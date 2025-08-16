from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from .models.base import PredictRequest, SessionIdManager
from .dependencies.inject_model import access_model_instance
from fastapi.responses import StreamingResponse, Response,JSONResponse
import threading, cv2, time
from .utils.core import truncate_history, build_prompt
from .utils.streamer import generate_stream,frame_memory
import numpy as np
import queue
from PIL import Image
from .utils.config import Config
from websockets import ConnectionClosedError

router = APIRouter()

MAX_FRAMES = 10  # FIFO max frames


@router.get("/ping")
def ping():
    return {"status": "ok"}


# @router.post("/generate")
#  # this function is to work with guff
# async def generate(request: PredictRequest, access=Depends(access_model_instance)):
#     session_id = request.session_id
#     prompt = request.prompt
#     model = access.get("model", "")

#     if not session_id or session_id not in frame_memory:
#         return JSONResponse({"error": "Invalid session_id or session not active"}, status_code=400)

#     q = frame_memory[session_id]["queue"]

#     # Collect frames safely
#     frames = []
#     with frame_memory[session_id]["lock"]:
#         while not q.empty():
#             frame_info = q.get()
#             frames.append(frame_info["frame"] if isinstance(frame_info, dict) else frame_info)

#     if not frames:
#         print(f"[Session {session_id}] No frames available in queue")
#         return JSONResponse({"error": "No frames to process"}, status_code=400)

#     print(f"Generate called with session_id: {session_id}, frames collected: {len(frames[-3:])}")

#     def token_generator():
#         for token in generate_stream(model, prompt, frame_queue=frames[-3:]):
#             yield token

#     return StreamingResponse(token_generator(), media_type="text/plain")

@router.post("/generate")
async def generate(request: PredictRequest, access=Depends(access_model_instance)):
    try:
        session_id = request.session_id
        prompt = request.prompt
        max_tokens = request.max_tokens
        access_model = access.get("model")
        model = access_model

        if not session_id or session_id not in frame_memory:
            return JSONResponse({"error": "Invalid session_id or session not active"}, status_code=400)

        q = frame_memory[session_id]["queue"]

        # Safely collect latest frames
        frames = []
        with frame_memory[session_id]["lock"]:
            frame_count = 0
            while not q.empty() and frame_count < 3:
                try:
                    item = q.get_nowait()
                    frames.append(item["frame"] if isinstance(item, dict) else item)
                    frame_count += 1
                except queue.Empty:
                    break

        def token_generator():
            stop_event = frame_memory[session_id]["stop_event"]
            output_q = queue.Queue()

            def _generate():
                try:
                    pil_imgs = []
                    if frames:
                        for f in frames:
                            if isinstance(f, np.ndarray):
                                pil_imgs.append(Image.fromarray(f))

                    # Generate tokens
                    for token in generate_stream(model, prompt, frame_queue=pil_imgs, stop_event=stop_event):
                        if stop_event.is_set():
                            break
                        output_q.put(token)

                except Exception as e:
                    output_q.put(f"[Error during generation]: {str(e)}")
                finally:
                    # Signal end of stream
                    output_q.put(None)

            threading.Thread(target=_generate, daemon=True).start()

            while True:
                token = output_q.get()
                if token is None or stop_event.is_set():
                    break
                yield token


        return StreamingResponse(token_generator(), media_type="text/plain")

    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {str(e)}"}, status_code=500)



@router.websocket("/stream_frame/{session_id}")
async def stream_frame(ws: WebSocket, session_id: str):
    session_id = session_id.strip()
    await ws.accept()
    
    # Initialize session memory
    frame_memory.setdefault(session_id, {
        "queue": queue.Queue(maxsize=MAX_FRAMES),
        "lock": threading.Lock(),
        "stop_event": threading.Event()
    })

    print(f"[{session_id}] WebSocket connection established.")

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                print(f"[{session_id}] Client disconnected.")
                break

            if "bytes" in msg and msg["bytes"] is not None:
                # Process frame data
                try:
                    nparr = np.frombuffer(msg["bytes"], np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is None:
                        print(f"[{session_id}] Failed to decode frame")
                        continue

                    # Resize frame for processing
                    frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                    
                    # Convert BGR to RGB for PIL compatibility
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                    # Store frame safely
                    q = frame_memory[session_id]["queue"]
                    with frame_memory[session_id]["lock"]:
                        if q.full():
                            # Remove oldest frame if queue is full
                            try:
                                _ = q.get_nowait()
                            except queue.Empty:
                                pass
                        q.put_nowait({"frame": frame_rgb, "timestamp": time.time()})

                except Exception as e:
                    print(f"[{session_id}] Frame processing error: {str(e)}")

            elif "text" in msg and msg["text"] is not None:
                # Handle text messages (heartbeat, control messages, etc.)
                print(f"[{session_id}] Received text message: {msg['text']}")
                
                # Optional: Send acknowledgment
                if msg["text"] == "ping":
                    await ws.send_text("pong")

    except WebSocketDisconnect:
        print(f"[{session_id}] WebSocket disconnected gracefully.")
    except Exception as e:
        print(f"[{session_id}] WebSocket error: {str(e)}")
    finally:
        # Clean up session
        if session_id in frame_memory:
            del frame_memory[session_id]
        print(f"[{session_id}] Session cleaned up.")



@router.post("/clear_session")
async def clear_session(request: SessionIdManager, access=Depends(access_model_instance)):
    session_id = request.session_id.strip()
    session_memory = access["memory"]
    running_gens = access["sessions"]
    frame_memory = access["frame_memory"]

    if session_id in frame_memory:
        stop_event = frame_memory[session_id].get("stop_event")
        if stop_event:
            stop_event.set()

    session_memory.pop(session_id, None)
    running_gens.pop(session_id, None)
    frame_memory.pop(session_id, None)

    return Response(f"{session_id} Deleted Successfully", status_code=200)


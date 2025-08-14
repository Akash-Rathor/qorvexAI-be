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
    session_id = request.session_id
    prompt = request.prompt
    model = access.get("model")

    if not session_id or session_id not in frame_memory:
        return JSONResponse({"error": "Invalid session_id or session not active"}, status_code=400)

    q = frame_memory[session_id]["queue"]

    # Collect latest frames
    frames = []
    with frame_memory[session_id]["lock"]:
        while not q.empty():
            item = q.get()
            frames.append(item["frame"] if isinstance(item, dict) else item)

    frames = frames[-3:] if frames else None  # Last 3 frames

    def token_generator():
        for token in model.generate_stream(prompt, frames=frames, max_tokens=500):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")





@router.websocket("/stream_frame/{session_id}")
async def stream_frame(ws: WebSocket, session_id: str):
    session_id = session_id.strip()
    await ws.accept()
    
    frame_memory.setdefault(session_id, {
        "queue": queue.Queue(maxsize=MAX_FRAMES),
        "lock": threading.Lock()
    })

    print(f"[{session_id}] WebSocket connection established.")

    try:
        while True:
            # Wait for incoming message (text or binary)
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                print(f"[{session_id}] Client disconnected.")
                break

            if "bytes" in msg and msg["bytes"] is not None:
                # Frame data
                nparr = np.frombuffer(msg["bytes"], np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f"[{session_id}] Failed to decode frame")
                    continue

                # Resize to avoid heavy processing
                frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

                # Store frame
                q = frame_memory[session_id]["queue"]
                with frame_memory[session_id]["lock"]:
                    if q.full():
                        _ = q.get_nowait()
                    q.put_nowait({"frame": frame_resized, "timestamp": time.time()})
                    # print(f"[{session_id}] Frame stored. Queue size: {q.qsize()}")

                # Optional: ack
                # await ws.send_text("frame_received")

            elif "text" in msg and msg["text"] is not None:
                print(f"[{session_id}] Received text message: {msg['text']}")
                # Handle heartbeats or control messages

    except WebSocketDisconnect:
        print(f"[{session_id}] WebSocket disconnected gracefully.")
    except ConnectionClosedError as e:
        print(f"[{session_id}] Connection closed unexpectedly: {e}")
    except Exception as e:
        print(f"[{session_id}] Error: {str(e)}")
    finally:
        # Clean up
        if session_id in frame_memory:
            del frame_memory[session_id]
        print(f"[{session_id}] Session cleaned up.")



@router.post("/clear_session")
async def clear_session(request: SessionIdManager, access=Depends(access_model_instance)):
    session_id = request.session_id.strip()
    session_memory = access["memory"]
    running_gens = access["sessions"]
    frame_memory = access["frame_memory"]

    session_memory.pop(session_id, None)
    running_gens.pop(session_id, None)
    frame_memory.pop(session_id, None)

    return Response(f"{session_id} Deleted Successfully", status_code=200)

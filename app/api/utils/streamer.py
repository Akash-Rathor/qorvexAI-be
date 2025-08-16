import queue
import threading
import time
from PIL import Image
import numpy as np
from transformers import TextIteratorStreamer
import re


def preprocess_images(frames, size=(224, 224)):
    """Convert frames (numpy arrays or PIL) to resized PIL images"""
    pil_imgs = []
    for f in frames:
        if isinstance(f, np.ndarray):
            img = Image.fromarray(f)
        elif isinstance(f, Image.Image):
            img = f
        else:
            continue
        img = img.resize(size)
        pil_imgs.append(img)
    return pil_imgs

frame_memory = {}

def build_prompt(prompt, num_images=0):
    image_tokens = ""
    for _ in range(num_images):
        image_tokens += "<start_of_image>\n"

    return f"""
    <start_of_turn>user
    {image_tokens}
    \n{prompt}
    You are a helpful AI assistant.
    Your primary task is to carefully analyze any screen image that is shared and provide responses directly related to the content of that screen.
    If the user asks a question unrelated to the screen, you may respond to that question separately.
    Always respond in English unless the user specifically requests a different language.
    <end_of_turn>
    <start_of_turn>model
    """

def filter_queue(text_data):
    # Implement your filtering logic here
    unwanted_outputs = ("Always respond in English unless the user specifically requests a different language.\n    \n    model\n    ",)
    if text_data in unwanted_outputs:
        return None
    if text_data == "<end_of_turn>":
        return None

    for sent in unwanted_outputs:
        if text_data.endswith(sent):
            return None

    # result = re.sub(r'data:', '', text_data)

    return text_data

def generate_stream(model_obj, prompt, frame_queue=None, max_new_tokens=256):
    output_q = queue.Queue()

    def _generate():
        nonlocal frame_queue
        if isinstance(frame_queue, list):
            new_q = queue.Queue()
            for f in frame_queue:
                new_q.put(f)
            frame_queue = new_q

        # while frame_queue is None or frame_queue.empty():
        #     time.sleep(0.1)

        # Process initial frames
        with frame_queue.mutex:
            pil_imgs = list(frame_queue.queue)
            # for f in list(frame_queue.queue):
            #     if isinstance(f, np.ndarray):
            #         pil_imgs.append(Image.fromarray(f))
            #     elif isinstance(f, dict) and "frame" in f:
            #         pil_imgs.append(Image.fromarray(f["frame"]))

        # Prepare inputs
        pil_imgs = pil_imgs[-1:]
        prepared_prompt = build_prompt(prompt, num_images=len(pil_imgs))
        model_obj_input = model_obj.processor(
            text=prepared_prompt,
            images=pil_imgs if pil_imgs else None,
            return_tensors="pt",
            padding=True
        ).to(model_obj.device)  # FIX: move to MPS/CPU/CUDA

        # Streaming setup
        streamer = TextIteratorStreamer(model_obj.processor.tokenizer, skip_special_tokens=True)
        
        # Run generation in thread
        gen_thread = threading.Thread(
            target=lambda: model_obj.model.generate(
                **model_obj_input,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            ),
            daemon=True
        )
        gen_thread.start()

        # Yield streamed tokens
        for text_piece in streamer:
            text_piece = filter_queue(text_piece)
            if text_piece:
                # print(f"{text_piece}",end=" ")
                output_q.put(text_piece)

    threading.Thread(target=_generate, daemon=True).start()

    while True:
        token = output_q.get()
        if token is None:
            break
        yield token
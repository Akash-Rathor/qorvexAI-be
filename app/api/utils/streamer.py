import queue
import threading
import time
from PIL import Image
import numpy as np

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


def generate_stream(model, prompt, frame_queue=None, max_new_tokens=256):
    """
    Streams tokens from model using frames from frame_queue (queue.Queue or list of arrays).
    """
    output_q = queue.Queue()

    def _generate():
        nonlocal frame_queue
        # Convert list to queue if needed
        if isinstance(frame_queue, list):
            new_q = queue.Queue()
            for f in frame_queue:
                new_q.put(f)
            frame_queue = new_q

        # Wait for at least one frame
        while frame_queue is None or frame_queue.empty():
            time.sleep(0.1)

        # Process initial frames
        with frame_queue.mutex:
            pil_imgs = []
            for f in list(frame_queue.queue):
                if isinstance(f, np.ndarray):
                    pil_imgs.append(Image.fromarray(f))
                elif isinstance(f, dict) and "frame" in f:
                    pil_imgs.append(Image.fromarray(f["frame"]))

        # Initial generation
        try:
            for token in model.generate_multimodal(prompt, images=pil_imgs, max_new_tokens=max_new_tokens):
                output_q.put(token)
        except Exception as e:
            output_q.put(f"[Error during generation]: {e}")

        # Continuously process new frames
        while True:
            latest_frames = []
            while frame_queue and not frame_queue.empty():
                f = frame_queue.get()
                if isinstance(f, np.ndarray):
                    latest_frames.append(f)
                elif isinstance(f, dict) and "frame" in f:
                    latest_frames.append(f["frame"])
            if latest_frames:
                pil_imgs = [Image.fromarray(f) for f in latest_frames]
                try:
                    for token in model.generate_multimodal(prompt, images=pil_imgs, max_new_tokens=64):
                        output_q.put(token)
                except Exception as e:
                    output_q.put(f"[Error during generation]: {e}")
            time.sleep(0.3)

    threading.Thread(target=_generate, daemon=True).start()

    while True:
        token = output_q.get()
        if token is None:
            break
        yield token
# app/core/load_model.py
import os
import base64
import cv2
from typing import Iterable, List, Optional
from llama_cpp import Llama

class ModelWrapper:
    def __init__(self, model_dir: str = "models/gemma_guff", n_ctx: int = 4096):
        self.model_dir = model_dir
        self.n_ctx = n_ctx
        self.model: Optional[Llama] = None
        print("ModelWrapper initialized (GGUF model not loaded yet).")

    def load(self):
        model_path = os.path.join(self.model_dir, "gemma-3-12b-it-q4_0.gguf")
        mmproj_path = os.path.join(self.model_dir, "mmproj-model-f16-12B.gguf")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"mmproj not found at: {mmproj_path}")

        print(f"Loading GGUF model from {model_path}")

        self.model = Llama(
            model_path=model_path,
            chat_format="gemma",
            clip_model_path=mmproj_path,  
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,
            logits_all=False,
            vocab_only=False,
            verbose=False,
        )
        print("GGUF model loaded successfully.")
        return self

    @staticmethod
    def _frame_to_data_uri(frame_bgr) -> str:
        """Convert a BGR OpenCV frame to JPEG data URI"""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("Failed to encode frame to JPEG")
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def generate_stream(
        self,
        prompt: str,
        frames: Optional[List] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Iterable[str]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert frames to model-readable format
        content = [{"type": "text", "text": prompt}]
        if frames:
            for f in frames:
                content.append({"type": "image_url", "image_url": self._frame_to_data_uri(f)})

        messages = [{"role": "user", "content": content}]

        # Stream tokens
        stream = self.model.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<end_of_turn>", "<eos>"],
        )
        for chunk in stream:
            try:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    yield delta["content"]
            except Exception:
                continue

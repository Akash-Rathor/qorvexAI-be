# load_model.py
import queue
import threading
from transformers import TextIteratorStreamer
import time
from PIL import Image
import numpy as np


class Gemma3ModelWrapper:
    def __init__(self,model_path=None):
        self.model_path = model_path
        self.device_type = None
        self.device = None
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.frame_memory = {}
        print("ModelWrapper initialized (model not loaded yet).")

    def build_prompt(self, prompt, num_images=0):
        image_tokens = ""
        for _ in range(num_images):
            image_tokens += "<start_of_image>\n"

        return f"""
        <start_of_turn>user
        {image_tokens}
        \n{prompt}
        You are a helpful AI assistant.
        Your primary task is to carefully analyze user's message and any screen image that is shared and provide responses directly related to the context of user's message and screen context.
        prioritize user message and understand if user need details related to screen, only then analyse the screen and provide relevant information.
        If the user asks a question unrelated to the screen, you may respond to that question separately.
        Always respond short and concisely do not explain things much unless explicitly asked.
        Always respond in English unless the user specifically requests a different language.
        <end_of_turn>
        <start_of_turn>model
        """


    def filter_queue(self,text_data):
        # Implement your filtering logic here
        unwanted_outputs = ("Always respond short and concisely do not explain things much unless explicitly asked.\n        Always respond in English unless the user specifically requests a different language.\n        \n        model\n        ",)
        if text_data in unwanted_outputs:
            return None
        if text_data == "<end_of_turn>":
            return None

        for sent in unwanted_outputs:
            if text_data.endswith(sent):
                return None

        # result = re.sub(r'data:', '', text_data)

        return text_data

    def generate_stream(self,model_obj, prompt, frame_queue=None, stop_event=None, max_new_tokens=256):
        output_q = queue.Queue()

        def _generate():
            nonlocal frame_queue
            if isinstance(frame_queue, list):
                new_q = queue.Queue()
                for f in frame_queue:
                    new_q.put(f)
                frame_queue = new_q

            with frame_queue.mutex:
                pil_imgs = list(frame_queue.queue)

            pil_imgs = pil_imgs[-1:]
            prepared_prompt = self.build_prompt(prompt, num_images=len(pil_imgs))
            model_obj_input = model_obj.processor(
                text=prepared_prompt,
                images=pil_imgs if pil_imgs else None,
                return_tensors="pt",
                padding=True
            ).to(model_obj.device)

            streamer = TextIteratorStreamer(model_obj.processor.tokenizer, skip_special_tokens=True)

            def run_generation():
                model_obj.model.generate(
                    **model_obj_input,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer
                )

            threading.Thread(target=run_generation, daemon=True).start()

            for text_piece in streamer:
                if stop_event and stop_event.is_set():
                    break
                text_piece = self.filter_queue(text_piece)
                if text_piece:
                    output_q.put(text_piece)

            output_q.put(None)

        threading.Thread(target=_generate, daemon=True).start()

        while True:
            if stop_event and stop_event.is_set():
                break
            token = output_q.get()
            if token is None:
                break
            yield token


    def _detect_device(self):
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

        self.device_type = self._detect_device()
        print(f"Using device: {self.device_type}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,use_fast=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path,use_fast=True)

        try:
            if self.device_type == "mps":
                self.device = torch.device("mps")
                # torch.backends.mps.enable_mps_fallback(True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                ).to(self.device)
            elif self.device_type == "cuda":
                self.device = torch.device("cuda")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path
                ).to(self.device)
            else:
                self.device = torch.device("cpu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    low_cpu_mem_usage=True
                )
        except Exception as e:
            print(f"[MPS load failed, falling back to CPU]: {e}")
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True
            )

        finally:
            return self

    def clear_cache(self):
        import torch
        if self.device_type == "mps":
            torch.mps.empty_cache()
        elif self.device_type == "cuda":
            torch.cuda.empty_cache()

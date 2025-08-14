# load_model.py

class ModelWrapper:
    def __init__(self, model_path="models/gemma"):
        self.model_path = model_path
        self.device_type = None
        self.device = None
        self.model = None
        self.tokenizer = None
        self.processor = None
        print("ModelWrapper initialized (model not loaded yet).")

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        try:
            if self.device_type == "mps":
                self.device = torch.device("mps")
                torch.backends.mps.enable_mps_fallback(True)
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

    def clear_cache(self):
        import torch
        if self.device_type == "mps":
            torch.mps.empty_cache()
        elif self.device_type == "cuda":
            torch.cuda.empty_cache()

# load_model.py

from .all_models.gemma3 import Gemma3ModelWrapper
from .all_models.qwen import QwenModelWrapper

class ModelWrapper:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def load(self):
        if self.model_path == "models/gemma":
            model_instance = Gemma3ModelWrapper(self.model_path)
        elif self.model_path == "models/qwen":
            model_instance = QwenModelWrapper(self.model_path)
        else:
            raise ValueError(f"Unknown model path: {self.model_path}")

        return model_instance.load()
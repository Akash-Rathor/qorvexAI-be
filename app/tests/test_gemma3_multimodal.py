# test_gemma3_multimodal.py
from llama_cpp import Llama
import base64

import os

print(os.getcwd())

# Load model (adjust path if different)
llm = Llama(
    model_path="models/gemma_guff/gemma-3-12b-it-q4_0.gguf",  # relative to this file
    mmproj_path="models/gemma_guff/mmproj-model-f16-12B.gguf",
    n_ctx=4096,
    n_threads=8,      # adjust for your CPU
    n_gpu_layers=-1,  # use GPU acceleration
    chat_format="gemma"
)

# Load an image and convert to base64
with open("test_image.png", "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Create a simple text+image prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    }
]

# Run inference
output = llm.create_chat_completion(messages=messages, temperature=0.2)

# Print model output
print("Model response:", output["choices"][0]["message"]["content"])


from typing import List, Dict
from ..models.base import Message
import time

MAX_TOKENS = 10000
TRUNCATE_TO = 10000



def approx_token_len(text: str) -> int:
    # Simple approximation: 1 token ~ 4 chars
    return len(text) // 4

def truncate_history(history: List[Message]) -> List[Message]:
    # Build a prompt string and check token length
    while True:
        prompt = build_prompt(history)
        if approx_token_len(prompt) < MAX_TOKENS:
            return history
        # Remove oldest message (usually a user or assistant message)
        if history:
            history.pop(0)
        else:
            break
    return history


def build_prompt(history: List[Dict], visual_context: str = "") -> str:
    lines = ["system: You are a helpful assistant. Always respond in English."]

    for msg in history:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg["timestamp"]))
        role = msg["role"]
        if role == "user":
            lines.append(f"[{ts}] user: {msg['content']}")
        elif role == "assistant":
            lines.append(f"[{ts}] assistant (previous reply): {msg['content']}")

    if visual_context:
        lines.append(f"[Visual context]: {visual_context}")

    lines.append("assistant: ")
    return "\n".join(lines)



from fastapi import Request

def access_model_instance(request: Request):
    return {
        "model": request.app.state.model_instance,
        "sessions": request.app.state.running_generations,
        "memory": request.app.state.session_memory,
        "frame_memory": request.app.state.frame_memory
    }

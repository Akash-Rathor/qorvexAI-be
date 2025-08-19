from fastapi import FastAPI
from app.api.routes import router
from app.core import load_model as load_model
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "models/gemma"
    app.state.model_instance = load_model.ModelWrapper(model_path).load()
    app.state.running_generations = {}
    app.state.session_memory = {}
    app.state.frame_memory = {}
    yield
    print("🚪 Shutting down, unloading model.")

app = FastAPI(
    title="Ai Peer Programmer",
    description="An AI peer programmer, developed by Akash Rathor",
    version="1.0",
    lifespan=lifespan
)

origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow GET, POST, OPTIONS etc.
    allow_headers=["*"],
)


# Register API routes
app.include_router(router)

"""
DeepFake Sentinel - FastAPI Backend Application
Main entry point for the API server.
"""

import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
<<<<<<< HEAD
from fastapi.staticfiles import StaticFiles
=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
from contextlib import asynccontextmanager
from loguru import logger
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from app.api.router import api_router
from app.core.config import settings
from app.ml.model_manager import ModelManager


model_manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global model_manager

    logger.info("Starting DeepFake Sentinel API...")

    model_manager = ModelManager()
    await model_manager.load_models()

    app.state.model_manager = model_manager

    logger.info("Models loaded successfully")

    yield

    logger.info("Shutting down DeepFake Sentinel API...")
    if model_manager:
        await model_manager.unload_models()


app = FastAPI(
    title="DeepFake Sentinel API",
    description="AI-powered deepfake video detection system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

<<<<<<< HEAD
=======

>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
app.include_router(api_router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="static")
=======

app.include_router(api_router, prefix="/api/v1")
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b


@app.get("/")
async def root():
    return {"name": "DeepFake Sentinel", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
<<<<<<< HEAD
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
=======
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
<<<<<<< HEAD
        app, host="0.0.0.0", port=8000, reload=False, log_level="info"
=======
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
    )

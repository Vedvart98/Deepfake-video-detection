"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "DeepFake Sentinel"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    UPLOAD_DIR: Path = Path("data/uploads")
    PROCESSED_DIR: Path = Path("data/processed")
    MODEL_DIR: Path = Path("models")

    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024
    ALLOWED_VIDEO_TYPES: List[str] = [".mp4", ".avi", ".mov", ".webm"]

    VIDEO_FPS: float = 5.0
    FACE_CONFIDENCE_THRESHOLD: float = 0.3
    CLASSIFICATION_THRESHOLD: float = 0.7

    DEVICE: str = "cuda"
    BATCH_SIZE: int = 32

    REDIS_URL: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


for directory in [settings.UPLOAD_DIR, settings.PROCESSED_DIR, settings.MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

"""
API Router Configuration
"""

from fastapi import APIRouter
from app.api.endpoints import video, health, model

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(model.router, tags=["Model"])
api_router.include_router(video.router, tags=["Video Analysis"])

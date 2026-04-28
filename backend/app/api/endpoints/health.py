"""
Health check endpoints.
"""

from fastapi import APIRouter
import torch
from app.models.schemas import HealthResponseSchema

router = APIRouter()


@router.get("/health", response_model=HealthResponseSchema)
async def health_check():
    return HealthResponseSchema(
        status="healthy",
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )

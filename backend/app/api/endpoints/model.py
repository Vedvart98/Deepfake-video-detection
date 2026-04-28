"""
Model information endpoints.
"""

import torch
from fastapi import APIRouter, Depends, Request
from app.models.schemas import ModelInfoSchema
from app.ml.model_manager import ModelManager

router = APIRouter()


@router.get("/model/info", response_model=ModelInfoSchema)
async def get_model_info(request: Request):
    model_manager: ModelManager = request.app.state.model_manager

    model_info = model_manager.get_model_info()

    return ModelInfoSchema(
        name=model_info.get("name", "EfficientNet"),
        version=model_info.get("version", "1.0.0"),
        input_size=model_info.get("input_size", 224),
        num_classes=model_info.get("num_classes", 2),
        device=model_info.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        cuda_available=torch.cuda.is_available(),
    )

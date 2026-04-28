"""
Pydantic schemas for request/response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BoundingBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    width: int
    height: int


class FramePredictionSchema(BaseModel):
    frame_idx: int
    timestamp: float
    label: str
    confidence: float
    face_boxes: List[BoundingBoxSchema]


class VideoPredictionSchema(BaseModel):
    video_id: str
    label: str
    confidence: float
    confidence_level: str
    fake_score: float
    aggregation_method: str
    frame_predictions: List[FramePredictionSchema]


class AudioVisualSchema(BaseModel):
    sync_score: float
    is_desynced: bool
    confidence: float
    temporal_alignment: List[Dict[str, Any]]


class GradCAMSchema(BaseModel):
    frame_idx: int
    timestamp: float
    heatmap_url: str
    overlay_url: str
    manipulation_regions: List[Dict[str, Any]]


class AnalysisResponseSchema(BaseModel):
    job_id: str
    status: JobStatus
    video_id: Optional[str] = None
    video_prediction: Optional[VideoPredictionSchema] = None
    audio_visual: Optional[AudioVisualSchema] = None
    gradcam_visualizations: Optional[List[GradCAMSchema]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


class AnalysisRequestSchema(BaseModel):
    video_id: str
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelInfoSchema(BaseModel):
    name: str
    version: str
    input_size: int
    num_classes: int
    device: str
    cuda_available: bool


class HealthResponseSchema(BaseModel):
    status: str
    cuda_available: bool
    cuda_device_count: int

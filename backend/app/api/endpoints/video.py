"""
Video analysis endpoints.
"""
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from loguru import logger
import uuid
from app.models.schemas import (
    AnalysisResponseSchema,
    JobStatus,
    VideoPredictionSchema,
    FramePredictionSchema,
    BoundingBoxSchema,
    AudioVisualSchema,
    GradCAMSchema,
)
from app.ml.model_manager import ModelManager
from app.services.video_processor import VideoProcessor
from app.services.face_detector import FaceDetector
from app.services.classifier import DeepfakeClassifier, AggregationMethod
from app.core.config import settings

def predict_verdict(fake_prob):
    """Determine video verdict based on fake probability threshold.
    
    Only considers FAKE if fake_prob > 60%, otherwise REAL.
    """
    real_prob = 1 - fake_prob
    margin = abs(fake_prob - real_prob)

    if fake_prob > 0.70 and margin > 0.30:
        return "FAKE", "HIGH CONFIDENCE", fake_prob * 100
    elif fake_prob > 0.60:
        return "FAKE", "LOW CONFIDENCE", fake_prob * 100
    elif fake_prob > 0.40:
        return "UNCERTAIN", "REVIEW NEEDED", 50.0
    elif fake_prob > 0.30:
        return "REAL", "LOW CONFIDENCE", real_prob * 100
    else:
        return "REAL", "HIGH CONFIDENCE", real_prob * 100


router = APIRouter()

jobs = {}

@router.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: Request = None,
):
    job_id = str(uuid.uuid4())

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Security: Use Path to safely extract extension and filename
    file_path = Path(file.filename)
    ext = file_path.suffix.lower() if file_path.suffix else ""
    if ext not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_VIDEO_TYPES}",
        )

    upload_path = settings.UPLOAD_DIR / f"{job_id}{ext}"

    # Security: Write file in chunks to avoid loading entire file into memory
    with open(upload_path, "wb") as f:
        chunk_size = 1024 * 1024  # 1MB chunks
        total_size = 0
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > settings.MAX_UPLOAD_SIZE:
                # Clean up partial upload
                f.close()
                upload_path.unlink()
                raise HTTPException(status_code=400, detail="File too large")
            f.write(chunk)

    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "video_path": str(upload_path),
        "video_id": job_id,
    }

    background_tasks.add_task(process_video_task, job_id, upload_path, request)

    return {"job_id": job_id, "status": JobStatus.PENDING}


async def process_video_task(job_id: str, video_path: Path, request: Request):
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING

        model_manager = request.app.state.model_manager

        video_processor = VideoProcessor(fps=settings.VIDEO_FPS)

        processor_config = getattr(model_manager, "_processor_config", None)
        face_detector = FaceDetector(
            confidence_threshold=settings.FACE_CONFIDENCE_THRESHOLD,
            target_size=processor_config.get("image_size", {}).get("height", 224)
            if processor_config
            else 224,
            model_config=processor_config,
        )

        metadata = video_processor.get_metadata(video_path)

        frame_predictions = []
        face_predictions = []
        gradcam_visualizations = []

        target_num_frames = 16
        face_tensors_batch = []
        face_frames_info = []

        for frame in video_processor.extract_frames(video_path):
            detection = face_detector.detect_faces(
                frame.data, frame.frame_idx, frame.timestamp
            )

            if detection.has_faces:
                best_face = detection.get_best_face()
                if best_face:
                    face_crop = face_detector.extract_face(frame.data, best_face)
                    face_tensor = face_detector.preprocess_for_model(face_crop)

                    face_tensors_batch.append(face_tensor)
                    face_frames_info.append({"frame": frame, "face": best_face})

                    if len(face_tensors_batch) >= target_num_frames:
                        break

        if not face_tensors_batch:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = "No faces detected"
            return

        import torch
        import numpy as np

        if face_tensors_batch:
            types = [type(x).__name__ for x in face_tensors_batch[:3]]
            logger.info(f"Batch types: {types}")
            face_tensors_batch = np.array(face_tensors_batch, dtype=np.float32)

        face_tensor = torch.tensor(face_tensors_batch, dtype=torch.float32)

        label, confidence = model_manager.predict(face_tensor)

        label = label.upper()

        probs = model_manager._get_probabilities(face_tensor)
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        
        logger.info(f"=== MODEL OUTPUT ===")
        logger.info(f"Raw probs: fake={fake_prob:.4f}, real={real_prob:.4f}")
        logger.info(f"Face tensors shape: {face_tensor.shape}")

        for i, info in enumerate(face_frames_info):
            frame = info["frame"]
            best_face = info["face"]

            frame_label = "FAKE" if fake_prob >= 0.65 else "REAL"
            frame_conf = fake_prob if frame_label == "FAKE" else real_prob

            frame_pred = FramePredictionSchema(
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
                label=frame_label,
                confidence=frame_conf,
                face_boxes=[
                    BoundingBoxSchema(
                        x1=best_face.x1,
                        y1=best_face.y1,
                        x2=best_face.x2,
                        y2=best_face.y2,
                        confidence=best_face.confidence,
                        width=best_face.width,
                        height=best_face.height,
                    )
                ],
            )
            frame_predictions.append(frame_pred)

        if not frame_predictions:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = "No faces detected"
            return

        fake_scores = [fp.confidence for fp in frame_predictions]
        video_fake_score = sum(fake_scores) / len(fake_scores)
        
        logger.info(f"Video fake score: {video_fake_score:.4f}")
        
        video_label, confidence_level, video_confidence = predict_verdict(video_fake_score)
        
        logger.info(f"Final verdict: {video_label} ({confidence_level}) confidence={video_confidence:.2f}")

        video_prediction = VideoPredictionSchema(
            video_id=job_id,
            label=video_label,
            confidence=video_confidence,
            confidence_level=confidence_level,
            fake_score=video_fake_score,
            aggregation_method=AggregationMethod.AVERAGE.value,
            frame_predictions=frame_predictions,
        )

        audio_visual = AudioVisualSchema(
            sync_score=0.5, is_desynced=False, confidence=0.0, temporal_alignment=[]
        )

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["video_prediction"] = video_prediction
        jobs[job_id]["audio_visual"] = audio_visual
        jobs[job_id]["gradcam_visualizations"] = gradcam_visualizations

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    response = AnalysisResponseSchema(job_id=job_id, status=job["status"])

    if job["status"] == JobStatus.COMPLETED:
        response.video_id = job.get("video_id")
        response.video_prediction = job.get("video_prediction")
        response.audio_visual = job.get("audio_visual")
        response.gradcam_visualizations = job.get("gradcam_visualizations")
    elif job["status"] == JobStatus.FAILED:
        response.error_message = job.get("error", "Unknown error")

    return response


@router.delete("/result/{job_id}")
async def delete_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs.pop(job_id)

    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        Path(video_path).unlink()

    return {"message": "Job deleted"}

<<<<<<< HEAD
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger
import uuid
import time
=======
"""
Video analysis endpoints.
"""
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from loguru import logger
import uuid
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
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
<<<<<<< HEAD
from app.services.gradcam import GradCAMExplainer
from app.services.audio_visual import AudioVisualConsistency
from app.core.config import settings


def predict_verdict(fake_prob):
=======
from app.services.classifier import DeepfakeClassifier, AggregationMethod
from app.core.config import settings

def predict_verdict(fake_prob):
    """Determine video verdict based on fake probability threshold.
    
    Only considers FAKE if fake_prob > 60%, otherwise REAL.
    """
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
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

<<<<<<< HEAD

=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
@router.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: Request = None,
):
    job_id = str(uuid.uuid4())

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

<<<<<<< HEAD
=======
    # Security: Use Path to safely extract extension and filename
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
    file_path = Path(file.filename)
    ext = file_path.suffix.lower() if file_path.suffix else ""
    if ext not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_VIDEO_TYPES}",
        )

    upload_path = settings.UPLOAD_DIR / f"{job_id}{ext}"

<<<<<<< HEAD
    with open(upload_path, "wb") as f:
        chunk_size = 1024 * 1024
=======
    # Security: Write file in chunks to avoid loading entire file into memory
    with open(upload_path, "wb") as f:
        chunk_size = 1024 * 1024  # 1MB chunks
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
        total_size = 0
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > settings.MAX_UPLOAD_SIZE:
<<<<<<< HEAD
=======
                # Clean up partial upload
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
                f.close()
                upload_path.unlink()
                raise HTTPException(status_code=400, detail="File too large")
            f.write(chunk)

    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "video_path": str(upload_path),
        "video_id": job_id,
    }

<<<<<<< HEAD
    model_manager = request.app.state.model_manager
    background_tasks.add_task(process_video_task, job_id, upload_path, model_manager)
=======
    background_tasks.add_task(process_video_task, job_id, upload_path, request)
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

    return {"job_id": job_id, "status": JobStatus.PENDING}


<<<<<<< HEAD
async def process_video_task(job_id: str, video_path: Path, model_manager):
    start_time = time.time()
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING

=======
async def process_video_task(job_id: str, video_path: Path, request: Request):
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING

        model_manager = request.app.state.model_manager

>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
        video_processor = VideoProcessor(fps=settings.VIDEO_FPS)

        processor_config = getattr(model_manager, "_processor_config", None)
        face_detector = FaceDetector(
            confidence_threshold=settings.FACE_CONFIDENCE_THRESHOLD,
            target_size=processor_config.get("image_size", {}).get("height", 224)
            if processor_config
            else 224,
            model_config=processor_config,
        )

<<<<<<< HEAD
        gradcam_explainer = GradCAMExplainer(model_manager.model)
        audio_visual_analyzer = AudioVisualConsistency(device=str(model_manager.device))

=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
        metadata = video_processor.get_metadata(video_path)

        frame_predictions = []
        face_predictions = []
        gradcam_visualizations = []

        target_num_frames = 16
        face_tensors_batch = []
        face_frames_info = []

        for frame in video_processor.extract_frames(video_path):
<<<<<<< HEAD
            detection = await face_detector.detect_faces(
=======
            detection = face_detector.detect_faces(
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
                frame.data, frame.frame_idx, frame.timestamp
            )

            if detection.has_faces:
                best_face = detection.get_best_face()
                if best_face:
                    face_crop = face_detector.extract_face(frame.data, best_face)
                    face_tensor = face_detector.preprocess_for_model(face_crop)

                    face_tensors_batch.append(face_tensor)
<<<<<<< HEAD
                    face_frames_info.append({"frame": frame, "face": best_face, "crop": face_crop})
=======
                    face_frames_info.append({"frame": frame, "face": best_face})
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

                    if len(face_tensors_batch) >= target_num_frames:
                        break

        if not face_tensors_batch:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = "No faces detected"
            return

        import torch
        import numpy as np

<<<<<<< HEAD
        face_tensors_batch = np.array(face_tensors_batch, dtype=np.float32)
=======
        if face_tensors_batch:
            types = [type(x).__name__ for x in face_tensors_batch[:3]]
            logger.info(f"Batch types: {types}")
            face_tensors_batch = np.array(face_tensors_batch, dtype=np.float32)

>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
        face_tensor = torch.tensor(face_tensors_batch, dtype=torch.float32)

        label, confidence = model_manager.predict(face_tensor)

        label = label.upper()

        probs = model_manager._get_probabilities(face_tensor)
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
<<<<<<< HEAD
=======
        
        logger.info(f"=== MODEL OUTPUT ===")
        logger.info(f"Raw probs: fake={fake_prob:.4f}, real={real_prob:.4f}")
        logger.info(f"Face tensors shape: {face_tensor.shape}")
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

        for i, info in enumerate(face_frames_info):
            frame = info["frame"]
            best_face = info["face"]
<<<<<<< HEAD
            face_crop = info["crop"]
=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

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

<<<<<<< HEAD
            if gradcam_explainer.cam is not None:
                try:
                    face_tensor_single = face_detector.preprocess_for_model(face_crop)
                    face_tensor_single = torch.tensor(face_tensor_single).unsqueeze(0)

                    heatmap = gradcam_explainer.generate_heatmap(face_tensor_single, target_category=0)
                    overlay = gradcam_explainer.overlay_heatmap(face_crop, heatmap)

                    heatmap_path = settings.UPLOAD_DIR / f"{job_id}_gradcam_{frame.frame_idx}.png"
                    import cv2
                    cv2.imwrite(str(heatmap_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                    regions = gradcam_explainer.get_manipulation_regions(heatmap)

                    gradcam_visualizations.append(
                        GradCAMSchema(
                            frame_idx=frame.frame_idx,
                            timestamp=frame.timestamp,
                            heatmap_url=f"/api/v1/results/{job_id}/gradcam/{frame.frame_idx}",
                            overlay_url=f"/api/v1/results/{job_id}/gradcam/{frame.frame_idx}/overlay",
                            manipulation_regions=[{"x": r[0], "y": r[1], "width": r[2], "height": r[3]} for r in regions],
                        )
                    )
                except Exception as e:
                    logger.warning(f"GradCAM failed for frame {frame.frame_idx}: {e}")

=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
        if not frame_predictions:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = "No faces detected"
            return

        fake_scores = [fp.confidence for fp in frame_predictions]
        video_fake_score = sum(fake_scores) / len(fake_scores)
<<<<<<< HEAD

        video_label, confidence_level, video_confidence = predict_verdict(video_fake_score)
=======
        
        logger.info(f"Video fake score: {video_fake_score:.4f}")
        
        video_label, confidence_level, video_confidence = predict_verdict(video_fake_score)
        
        logger.info(f"Final verdict: {video_label} ({confidence_level}) confidence={video_confidence:.2f}")
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

        video_prediction = VideoPredictionSchema(
            video_id=job_id,
            label=video_label,
            confidence=video_confidence,
            confidence_level=confidence_level,
            fake_score=video_fake_score,
<<<<<<< HEAD
            aggregation_method="average",
            frame_predictions=frame_predictions,
        )

        audio_visual = None
        try:
            face_crops = [info["crop"] for info in face_frames_info]
            av_result = await audio_visual_analyzer.analyze_video(
                video_path, face_crops, metadata.fps
            )

            audio_visual = AudioVisualSchema(
                sync_score=av_result.sync_score,
                is_desynced=av_result.is_desynced,
                confidence=av_result.confidence,
                temporal_alignment=av_result.temporal_alignment,
            )
        except Exception as e:
            logger.warning(f"Audio-Visual analysis failed: {e}")
            audio_visual = AudioVisualSchema(
                sync_score=0.5, is_desynced=False, confidence=0.0, temporal_alignment=[]
            )

        processing_time = time.time() - start_time
=======
            aggregation_method=AggregationMethod.AVERAGE.value,
            frame_predictions=frame_predictions,
        )

        audio_visual = AudioVisualSchema(
            sync_score=0.5, is_desynced=False, confidence=0.0, temporal_alignment=[]
        )
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["video_prediction"] = video_prediction
        jobs[job_id]["audio_visual"] = audio_visual
        jobs[job_id]["gradcam_visualizations"] = gradcam_visualizations
<<<<<<< HEAD
        jobs[job_id]["processing_time"] = processing_time

        logger.info(f"Video analysis completed in {processing_time:.2f}s")
=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

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
<<<<<<< HEAD
        response.processing_time = job.get("processing_time")
=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
    elif job["status"] == JobStatus.FAILED:
        response.error_message = job.get("error", "Unknown error")

    return response


<<<<<<< HEAD
@router.get("/results/{job_id}/gradcam/{frame_idx}")
async def get_gradcam_heatmap(job_id: str, frame_idx: int):
    heatmap_path = settings.UPLOAD_DIR / f"{job_id}_gradcam_{frame_idx}.png"
    if not heatmap_path.exists():
        raise HTTPException(status_code=404, detail="GradCAM image not found")
    return FileResponse(str(heatmap_path), media_type="image/png")


@router.get("/results/{job_id}/gradcam/{frame_idx}/overlay")
async def get_gradcam_overlay(job_id: str, frame_idx: int):
    heatmap_path = settings.UPLOAD_DIR / f"{job_id}_gradcam_{frame_idx}.png"
    if not heatmap_path.exists():
        raise HTTPException(status_code=404, detail="GradCAM overlay not found")
    return FileResponse(str(heatmap_path), media_type="image/png")


=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
@router.delete("/result/{job_id}")
async def delete_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs.pop(job_id)

    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        Path(video_path).unlink()

<<<<<<< HEAD
    upload_dir = settings.UPLOAD_DIR
    for gradcam_file in upload_dir.glob(f"{job_id}_gradcam_*.png"):
        gradcam_file.unlink()

=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
    return {"message": "Job deleted"}

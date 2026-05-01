"""
Module 2: Face Detection using YOLOv8
Detects faces in video frames with bounding boxes and landmarks.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import cv2

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(
        "Ultralytics not installed. Face detection will use OpenCV fallback."
    )


@dataclass
class BoundingBox:
    """Face bounding box with confidence."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    landmarks: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "width": self.width,
            "height": self.height,
        }
        if self.landmarks is not None:
            result["landmarks"] = self.landmarks.tolist()
        return result


@dataclass
class FaceDetectionResult:
    """Result of face detection on a single frame."""

    frame_idx: int
    timestamp: float
    faces: List[BoundingBox] = field(default_factory=list)
    image_shape: Tuple[int, int, int] = (0, 0, 3)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0

    def get_largest_face(self) -> Optional[BoundingBox]:
        """Get the largest detected face."""
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.area)

    def get_best_face(self, quality_threshold: float = 0.5) -> Optional[BoundingBox]:
        """
        Get the best quality face based on size and confidence.

        Args:
            quality_threshold: Minimum confidence threshold

        Returns:
            Best face or None
        """
        valid_faces = [f for f in self.faces if f.confidence >= quality_threshold]
        if not valid_faces:
            return None

        # Score faces by area and confidence
        def score(face: BoundingBox) -> float:
            area_score = min(face.area / 10000, 1.0)  # Normalize area
            conf_score = face.confidence
            return 0.6 * area_score + 0.4 * conf_score

        return max(valid_faces, key=score)


class FaceDetector:
    """
    Face detector using YOLOv8.

    Features:
    - High accuracy face detection
    - 5-point facial landmarks
    - Face quality assessment
    - Batch processing support
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.7,
        iou_threshold: float = 0.45,
        margin: float = 0.2,
        target_size: int = 224,
        model_config: Optional[dict] = None,
    ):
        """
        Initialize face detector.

        Args:
            model_path: Path to YOLOv8 model (defaults to yolov8n-face)
            confidence_threshold: Minimum confidence for detection
            iou_threshold: NMS IoU threshold
            margin: Margin around face bounding box (fraction of size)
            target_size: Target size for extracted face crops
            model_config: Preprocessing config (from processor_config.json)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.margin = margin

        if model_config:
            img_size = model_config.get("image_size", {})
            self.target_size = (
                img_size.get("height", target_size)
                if isinstance(img_size, dict)
                else model_config.get("image_size", target_size)
            )
            self._mean = np.array(
                model_config.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32
            )
            self._std = np.array(
                model_config.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32
            )
            self._model_type = model_config.get("model_type", "imagenet")
        else:
            self.target_size = target_size
            self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            self._model_type = "imagenet"

<<<<<<< HEAD
        self._model_path = model_path
        self._yolo_model_path = None

        if ULTRALYTICS_AVAILABLE:
            model_dir = Path(__file__).parent.parent / "models"
            if model_path and Path(model_path).exists():
                self._yolo_model_path = Path(model_path)
            elif (model_dir / "yolov8n.pt").exists():
                self._yolo_model_path = model_dir / "yolov8n.pt"
            else:
                self._yolo_model_path = "yolov8n.pt"  # Will download on first use
            self.model = None  # Lazy-loaded
=======
        if ULTRALYTICS_AVAILABLE:
            if model_path and Path(model_path).exists():
                logger.info(f"Loading face detector from {model_path}")
                self.model = YOLO(str(model_path))
            else:
                # Use YOLOv8n from models directory
                model_dir = Path(__file__).parent.parent / "models"
                yolo_model_path = model_dir / "yolov8n.pt"
                if yolo_model_path.exists():
                    logger.info(f"Loading YOLOv8n from {yolo_model_path}")
                    self.model = YOLO(str(yolo_model_path))
                else:
                    # Fallback: try to download standard yolov8n
                    logger.info("Loading YOLOv8n model (downloading if needed)")
                    self.model = YOLO("yolov8n.pt")
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
        else:
            self.model = None
            logger.warning("Using OpenCV fallback for face detection")

<<<<<<< HEAD
    async def ensure_model_loaded(self):
        """Lazy-load YOLOv8 model (non-blocking)."""
        if self.model is not None:
            return

        if ULTRALYTICS_AVAILABLE and self._yolo_model_path is not None:
            try:
                logger.info(f"Loading YOLOv8 model from {self._yolo_model_path}")
                self.model = YOLO(str(self._yolo_model_path))
                logger.info("YOLOv8 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 model: {e}")
                self.model = None

    async def detect_faces(
        self, frame: np.ndarray, frame_idx: int = 0, timestamp: float = 0.0
    ) -> FaceDetectionResult:
        """Detect faces in a single frame."""
        await self.ensure_model_loaded()
        
=======
    def detect_faces(
        self, frame: np.ndarray, frame_idx: int = 0, timestamp: float = 0.0
    ) -> FaceDetectionResult:
        """
        Detect faces in a single frame.

        Args:
            frame: Input frame (BGR format)
            frame_idx: Frame index
            timestamp: Frame timestamp in seconds

        Returns:
            FaceDetectionResult with detected faces
        """
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
        result = FaceDetectionResult(
            frame_idx=frame_idx, timestamp=timestamp, image_shape=frame.shape
        )

        if self.model is not None:
            # Use YOLOv8
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )[0]

            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()

                # Extract landmarks if available
                landmarks = None
                if results.keypoints is not None:
                    landmarks = results.keypoints.xy.cpu().numpy()

                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    x1, y1, x2, y2 = map(int, box)

                    # Add margin
                    h, w = frame.shape[:2]
                    dx = int((x2 - x1) * self.margin)
                    dy = int((y2 - y1) * self.margin)
                    x1 = max(0, x1 - dx)
                    y1 = max(0, y1 - dy)
                    x2 = min(w, x2 + dx)
                    y2 = min(h, y2 + dy)

                    face_landmarks = landmarks[i] if landmarks is not None else None

                    result.faces.append(
                        BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=float(conf),
                            landmarks=face_landmarks,
                        )
                    )
        else:
            # OpenCV fallback (simplified)
            result.faces = self._opencv_detect(frame)

        return result

    def _opencv_detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Fallback face detection using OpenCV Haar cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            outputRejectLevels=True,
        )

        result = []
        rects, _, confidences = faces

        for i, (x, y, w, h) in enumerate(rects):
            result.append(
                BoundingBox(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    confidence=float(confidences[i]) / 100
                    if confidences[i] > 1
                    else float(confidences[i]),
                )
            )

        return result

    def extract_face(
        self, frame: np.ndarray, face: BoundingBox, align: bool = True
    ) -> np.ndarray:
        """
        Extract and preprocess face crop.

        Args:
            frame: Source frame
            face: Face bounding box
            align: Whether to align face using landmarks

        Returns:
            Preprocessed face crop (224x224)
        """
        # Extract face region
        face_crop = frame[face.y1 : face.y2, face.x1 : face.x2].copy()

        if align and face.landmarks is not None:
            # Align face using eye landmarks
            face_crop = self._align_face(face_crop, face.landmarks, face)

        # Resize to target size
        face_crop = cv2.resize(face_crop, (self.target_size, self.target_size))

        return face_crop

    def _align_face(
        self, face_crop: np.ndarray, landmarks: np.ndarray, face: BoundingBox
    ) -> np.ndarray:
        """Align face using eye landmarks."""
        if len(landmarks) < 2:
            return face_crop

        # Extract eye positions relative to face crop
        left_eye = landmarks[0]  # Usually left eye
        right_eye = landmarks[1]  # Usually right eye

        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate center between eyes
        eyes_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        # Rotate face
        h, w = face_crop.shape[:2]
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(face_crop, M, (w, h))

        return aligned

    def preprocess_for_model(self, face_crop: np.ndarray) -> np.ndarray:
        """Preprocess face crop for model input."""
        if face_crop.ndim == 2:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        elif face_crop.shape[-1] == 1:
            face_crop = face_crop.squeeze(-1)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        elif face_crop.shape[-1] == 3:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        face_crop = cv2.resize(face_crop, (self.target_size, self.target_size))
        face_crop = face_crop.astype(np.float32) / 255.0
        face_crop = (face_crop - self._mean) / self._std

        face_crop = face_crop.transpose(2, 0, 1)
        return face_crop

    def detect_batch(
        self, frames: List[Tuple[np.ndarray, int, float]]
    ) -> List[FaceDetectionResult]:
        """
        Detect faces in a batch of frames.

        Args:
            frames: List of (frame, frame_idx, timestamp) tuples

        Returns:
            List of FaceDetectionResult
        """
        return [self.detect_faces(frame, idx, ts) for frame, idx, ts in frames]

    def __repr__(self) -> str:
        return (
            f"FaceDetector(conf={self.confidence_threshold}, "
            f"margin={self.margin}, target={self.target_size})"
        )

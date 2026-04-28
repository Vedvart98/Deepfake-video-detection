"""
Model Manager - handles ML model lifecycle for deepfake detection.
Loads fastai exported EfficientNet-B4 model - uses model directly for inference.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
from loguru import logger
import numpy as np
from PIL import Image

from app.core.config import settings


class ModelManager:
    """Manages ML model loading and inference."""

    def __init__(self, force_download: bool = False, resume_download: bool = True):
        self.device = torch.device(
            settings.DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[torch.nn.Module] = None
        self.learn: Optional[Any] = None
        self._model_info: Dict = {}
        self._classes: list = ["real", "fake"]

    async def load_models(self):
        from fastai.learner import load_learner

        logger.info(f"Loading fastai model on device: {self.device}")

        model_path = settings.MODEL_DIR / "deepfake_model-3.pkl"

        if not model_path.exists():
            logger.error(f"Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            self.learn = load_learner(model_path, cpu=self.device.type == "cpu")
            self.model = self.learn.model

            self.model = self.model.to(self.device)
            self.model.eval()

            if hasattr(self.learn, 'dls') and hasattr(self.learn.dls, 'c2i'):
                self._classes = list(self.learn.dls.c2i.keys())

            self._model_info = {
                "name": "EfficientNet-B4 (fastai)",
                "version": "1.0.0",
                "device": str(self.device),
                "model_path": str(model_path),
                "classes": self._classes,
            }

            logger.info(f"Loaded fastai model successfully from {model_path}")
            logger.info(f"Classes: {self._classes}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_models(self):
        if self.model:
            del self.model
            self.model = None
        if self.learn:
            del self.learn
            self.learn = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded")

    def predict(self, face_tensors: torch.Tensor) -> Tuple[str, float]:
        """
        Predict using the fastai model directly (not using learn.predict).
        
        Args:
            face_tensors: Image tensor [C, H, W] or [B, C, H, W]
        
        Returns:
            Tuple of (label: str, confidence: float)
            - label: "real" or "fake"
            - confidence: probability score (0-1)
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        return self._predict_torch(face_tensors)

    def _get_probabilities(self, face_tensors: torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model loaded")
            
        current_dim = face_tensors.dim()
        
        if current_dim == 5:
            batch_size, num_frames = face_tensors.shape[:2]
            face_tensors = face_tensors.view(batch_size * num_frames, *face_tensors.shape[2:])
        elif current_dim == 4:
            pass
        elif current_dim == 3:
            face_tensors = face_tensors.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape: {face_tensors.shape}")

        face_tensors = face_tensors.to(self.device)

        with torch.no_grad():
            outputs = self.model(face_tensors)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            probs = F.softmax(outputs, dim=1)

        if hasattr(probs, 'numpy'):
            probs = probs.numpy()
        else:
            probs = np.array(probs)

        avg_probs = probs.mean(axis=0)
        return avg_probs

    def _predict_torch(self, face_tensors: torch.Tensor) -> Tuple[str, float]:
        import numpy as np
        from PIL import Image

        current_dim = face_tensors.dim()
        if current_dim == 4:
            face_tensors = face_tensors[0:1]
        elif current_dim == 3:
            face_tensors = face_tensors.unsqueeze(0)
        elif current_dim == 5:
            face_tensors = face_tensors[0, 0:1]
        else:
            raise ValueError(f"Unexpected tensor shape: {face_tensors.shape}")

        face_np = face_tensors.squeeze(0).cpu().numpy()

        if face_np.max() <= 2.5 and face_np.min() >= -2.5:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            face_np = (face_np * std + mean)
            face_np = np.clip(face_np, 0, 1)
            face_np = (face_np * 255).astype(np.uint8)
        elif face_np.max() <= 1.0:
            face_np = (face_np * 255).astype(np.uint8)
        else:
            face_np = face_np.astype(np.uint8)

        if face_np.shape[0] == 3:
            face_np = face_np.transpose(1, 2, 0)

        face_img = Image.fromarray(face_np).resize((224, 224), Image.BILINEAR)

        img_array = np.array(face_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            probs = F.softmax(outputs, dim=1)

        if hasattr(probs, 'numpy'):
            probs = probs.numpy()
        else:
            probs = np.array(probs)

        pred_idx = int(probs[0].argmax())
        confidence = float(probs[0].max())

        label = self._classes[pred_idx] if pred_idx < len(self._classes) else ("fake" if pred_idx == 1 else "real")

        return label, confidence

    def predict_from_image(self, image_path: str) -> Tuple[str, float]:
        if self.model is None:
            raise RuntimeError("No model loaded")

        from PIL import Image
        import numpy as np

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            probs = F.softmax(outputs, dim=1)

        if hasattr(probs, 'numpy'):
            probs = probs.numpy()
        else:
            probs = np.array(probs)

        pred_idx = int(probs[0].argmax())
        confidence = float(probs[0].max())

        label = self._classes[pred_idx] if pred_idx < len(self._classes) else ("fake" if pred_idx == 1 else "real")

        return label, confidence

    def get_model_info(self) -> Dict:
        return self._model_info.copy()

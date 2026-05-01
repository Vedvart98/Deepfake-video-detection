"""Model Manager - handles ML model lifecycle for deepfake detection."""
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
from loguru import logger
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.feature_extractor import HCiTModel, ModelConfig


class ModelManager:
    def __init__(self, force_download: bool = False, resume_download: bool = True):
        self.device = torch.device(
            settings.DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[torch.nn.Module] = None
        self.learn: Optional[Any] = None
        self._model_info: Dict = {}
        self._classes: list = ["real", "fake"]
        self.model_type: str = "unknown"

    async def load_models(self):
        hcit_path = settings.MODEL_DIR / "hcit_model.pth"
        hcit_config_path = settings.MODEL_DIR / "hcit_config.json"

        if hcit_path.exists():
            try:
                await self._load_hcit_model(hcit_path, hcit_config_path)
                return
            except Exception as e:
                logger.warning(f"Failed to load HCiT model: {e}. Falling back to EfficientNet.")

        await self._load_fastai_model()

    async def _load_hcit_model(self, model_path: Path, config_path: Optional[Path] = None):
        config = None
        if config_path and config_path.exists():
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        else:
            config = ModelConfig()

        self.model = HCiTModel(config)
        self.model_type = "hcit"

        state_dict = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._classes = ["real", "fake"]
        self._model_info = {
            "name": "HCiT (Hybrid CNN + Vision Transformer)",
            "version": "1.0.0",
            "device": str(self.device),
            "model_path": str(model_path),
            "classes": self._classes,
            "architecture": "ResNet50 + DeiT-Small with Cross-Attention Fusion",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }

        logger.info(f"Loaded HCiT model successfully from {model_path}")

    async def _load_fastai_model(self):
        from fastai.learner import load_learner

        logger.info(f"Loading fastai model on device: {self.device}")

        model_path = settings.MODEL_DIR / "hcit_deepfake_model.pkl"

        if not model_path.exists():
            logger.error(f"Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.learn = load_learner(model_path, cpu=self.device.type == "cpu")
        self.model = self.learn.model
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = "efficientnet"

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
        if self.model is None:
            raise RuntimeError("No model loaded")

        if self.model_type == "hcit":
            return self._predict_hcit(face_tensors)
        return self._predict_fastai(face_tensors)

    def _predict_hcit(self, face_tensors: torch.Tensor) -> Tuple[str, float]:
        current_dim = face_tensors.dim()

        if current_dim == 5:
            batch_size, num_frames = face_tensors.shape[:2]
            face_tensors = face_tensors.reshape(batch_size * num_frames, *face_tensors.shape[2:])
            is_video = True
        elif current_dim == 4:
            batch_size = face_tensors.shape[0]
            is_video = False
        elif current_dim == 3:
            face_tensors = face_tensors.unsqueeze(0)
            batch_size = 1
            is_video = False
        else:
            raise ValueError(f"Unexpected tensor shape: {face_tensors.shape}")

        face_tensors = face_tensors.to(self.device)

        with torch.no_grad():
            if is_video:
                logits = self.model(face_tensors, aggregate=True)
            else:
                logits = self.model(face_tensors)

            if isinstance(logits, dict) and 'logits' in logits:
                logits = logits['logits']
            elif hasattr(logits, 'logits'):
                logits = logits.logits

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probs = F.softmax(logits, dim=1)

        if probs.dim() == 2 and probs.shape[0] > 1:
            avg_probs = probs.mean(dim=0)
            pred_idx = int(avg_probs.argmax())
            confidence = float(avg_probs.max())
        else:
            pred_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs.max(dim=1).values.item())

        label = self._classes[pred_idx] if pred_idx < len(self._classes) else ("fake" if pred_idx == 1 else "real")

        return label, confidence

    def _predict_fastai(self, face_tensors: torch.Tensor) -> Tuple[str, float]:
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

    def get_frame_predictions(self, face_tensors: torch.Tensor) -> List[Tuple[str, float, float]]:
        if self.model is None:
            raise RuntimeError("No model loaded")

        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)

        face_tensors = face_tensors.to(self.device)

        with torch.no_grad():
            logits = self.model(face_tensors)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            probs = F.softmax(logits, dim=1)

        results = []
        for i in range(probs.shape[0]):
            fake_prob = float(probs[i, 0])
            real_prob = float(probs[i, 1])
            label = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = max(fake_prob, real_prob)
            results.append((label, confidence, fake_prob))

        return results

    def get_model_info(self) -> Dict:
        return self._model_info.copy()

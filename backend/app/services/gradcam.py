"""
Module 5: Grad-CAM Explainability
Generates attention heatmaps to highlight manipulated facial regions.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import cv2
from loguru import logger

try:
    from pytorch_grad_cam import GradCAM, HiResCAM, AblationCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not available. Using fallback implementation.")


class GradCAMExplainer:
    """
    Generates Grad-CAM heatmaps for deepfake detection explanations.
    Supports CNN and Vision Transformer architectures.
    """

    def __init__(
        self, model: torch.nn.Module, target_layer_name: str = "vit.blocks[-1].norm1"
    ):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.cam = None
        self._setup_cam()

    def _setup_cam(self):
        """Initialize Grad-CAM with appropriate target layer."""
        if not GRADCAM_AVAILABLE:
            return

        try:
            target_layers = self._get_target_layers()
            self.cam = GradCAM(
                model=self.model,
                target_layers=target_layers,
                # use_cuda removed in pytorch-grad-cam 1.5+
                # Library auto-detects device from model
            )
        except Exception as e:
            logger.warning(f"Failed to setup Grad-CAM: {e}")
            self.cam = None

    def _get_target_layers(self) -> List[torch.nn.Module]:
        """Get target layers for Grad-CAM based on model architecture."""
        layers = []

        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                layers.append(module)

        if not layers:
            if hasattr(self.model, "cnn"):
                for name, module in self.model.cnn.named_modules():
                    if "layer4" in name and isinstance(module, torch.nn.Conv2d):
                        layers.append(module)

        if not layers and hasattr(self.model, "features"):
            layers.append(self.model.features[-1])

        return layers

    def _reshape_transform(self, activations: torch.Tensor) -> torch.Tensor:
        """Reshape ViT activations for Grad-CAM compatibility."""
        if activations.dim() == 3:
            batch_size, seq_len, channels = activations.shape
            height = width = int(np.sqrt(seq_len - 1))

            if height * width < seq_len - 1:
                activations = activations[:, 1:, :]

            activations = activations.reshape(batch_size, height, width, channels)
            activations = activations.transpose(0, 3)
            activations = activations.transpose(1, 2)

        return activations

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_category: int = 0,
        method: str = "gradcam",
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single input.

        Args:
            input_tensor: Input tensor [3, H, W] or [1, 3, H, W]
            target_category: Target class (0=FAKE, 1=REAL)
            method: CAM method ('gradcam', 'hirescam', 'ablationcam')

        Returns:
            Heatmap array [H, W] with values in [0, 1]
        """
        if self.cam is None or not GRADCAM_AVAILABLE:
            return self._fallback_heatmap(input_tensor)

        targets = [ClassifierOutputTarget(target_category)]

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()

        grayscale_cam = self.cam(
            input_tensor, targets=targets, eigen_smooth=True, aug_smooth=False
        )[0, :]

        return grayscale_cam

    def _fallback_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Fallback gradient-based attention when Grad-CAM is unavailable."""
        input_tensor.requires_grad_(True)

        outputs = self.model(input_tensor.unsqueeze(0))

        if outputs.dim() > 1:
            fake_score = outputs[0, 0]
        else:
            fake_score = outputs[0]

        self.model.zero_grad()
        fake_score.backward()

        gradients = input_tensor.grad
        if gradients is None:
            return np.zeros((input_tensor.shape[1], input_tensor.shape[2]))

        attention = torch.mean(torch.abs(gradients), dim=0)
        attention = attention.cpu().numpy()

        attention = attention - attention.min()
        if attention.max() > 0:
            attention = attention / attention.max()

        h, w = input_tensor.shape[1], input_tensor.shape[2]
        heatmap = cv2.resize(attention, (w, h))

        return heatmap

    def overlay_heatmap(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on original frame.

        Args:
            frame: Original frame [H, W, 3] in RGB
            heatmap: Heatmap [H, W] in [0, 1]
            alpha: Blending coefficient
            colormap: OpenCV colormap constant

        Returns:
            Blended image [H, W, 3]
        """
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        if heatmap.max() <= 1.0:
            heatmap = (heatmap * 255).astype(np.uint8)

        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

        return blended

    def generate_batch_heatmaps(
        self, input_tensors: torch.Tensor, target_category: int = 0
    ) -> List[np.ndarray]:
        """Generate heatmaps for a batch of inputs."""
        heatmaps = []

        for i in range(input_tensors.shape[0]):
            heatmap = self.generate_heatmap(input_tensors[i], target_category)
            heatmaps.append(heatmap)

        return heatmaps

    def visualize_attention(
        self, input_tensor: torch.Tensor, attention_weights: torch.Tensor
    ) -> np.ndarray:
        """
        Visualize attention weights from ViT.

        Args:
            input_tensor: Input image tensor
            attention_weights: Attention weights from ViT

        Returns:
            Attention visualization
        """
        if attention_weights.dim() == 4:
            attention = attention_weights[0].mean(dim=0)
        else:
            attention = attention_weights

        attention = attention.cpu().numpy()

        if attention.shape[0] > 224 * 224:
            num_patches = int(np.sqrt(attention.shape[0] - 1))
            attention = attention[1:].reshape(num_patches, num_patches)
        else:
            size = int(np.sqrt(attention.shape[0]))
            if size * size == attention.shape[0]:
                attention = attention.reshape(size, size)

        attention = attention - attention.min()
        if attention.max() > 0:
            attention = attention / attention.max()

        h, w = input_tensor.shape[1], input_tensor.shape[2]
        heatmap = cv2.resize(attention, (w, h))

        return heatmap

    def get_manipulation_regions(
        self, heatmap: np.ndarray, threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Identify manipulation regions from heatmap.

        Args:
            heatmap: Heatmap [H, W]
            threshold: Threshold for region detection

        Returns:
            List of bounding boxes (x, y, w, h) for high-attention regions
        """
        binary_map = (heatmap >= threshold).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:
                regions.append((x, y, w, h))

        return regions

    def generate_explanation_report(
        self, input_tensor: torch.Tensor, prediction: Tuple[str, float]
    ) -> dict:
        """
        Generate comprehensive explanation report.

        Args:
            input_tensor: Input face tensor
            prediction: (label, confidence) tuple

        Returns:
            Explanation report dictionary
        """
        heatmap = self.generate_heatmap(input_tensor, target_category=0)

        regions = self.get_manipulation_regions(heatmap)

        if hasattr(self.model, "get_attention_weights"):
            try:
                attention = self.model.get_attention_weights(input_tensor.unsqueeze(0))
                attn_viz = self.visualize_attention(input_tensor, attention)
            except:
                attn_viz = None
        else:
            attn_viz = None

        return {
            "prediction": {"label": prediction[0], "confidence": prediction[1]},
            "heatmap": {
                "values": heatmap.tolist(),
                "min": float(heatmap.min()),
                "max": float(heatmap.max()),
                "mean": float(heatmap.mean()),
            },
            "manipulation_regions": [
                {"x": r[0], "y": r[1], "width": r[2], "height": r[3]} for r in regions
            ],
            "attention_visualization": attn_viz.tolist()
            if attn_viz is not None
            else None,
            "num_suspicious_regions": len(regions),
        }

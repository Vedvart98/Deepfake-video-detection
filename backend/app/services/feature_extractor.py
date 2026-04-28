"""
Module 3: Feature Extraction using CNN + Vision Transformer Hybrid Architecture
Implements HCiT (Hybrid CNN and Image Transformer) for deepfake detection.
Based on: HCiT - Hybrid CNN and Image Transformer for Deepfake Detection (IEEE 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    from torchvision.models import resnet50
    from torchvision.models import ResNet50_Weights

    TRANSFORMERS_AVAILABLE = True
except (ImportError, AttributeError):
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Torchvision not available. Using CNN-only model.")


@dataclass
class ModelConfig:
    """Configuration for HCiT model."""

    # CNN settings
    cnn_model: str = "resnet50"
    cnn_pretrained: bool = True
    cnn_freeze_layers: int = 0  # Number of layers to freeze

    # ViT settings
    vit_model: str = "deit_small_patch16_224"
    vit_pretrained: bool = True
    vit_embed_dim: int = 384
    vit_num_heads: int = 6
    vit_depth: int = 12
    vit_mlp_ratio: float = 4.0
    vit_drop_rate: float = 0.0
    vit_attn_drop_rate: float = 0.0

    # Fusion settings
    fusion_hidden_dim: int = 1024
    dropout: float = 0.3

    # Classification
    num_classes: int = 2
    output_dim: int = 512


class CNNBranch(nn.Module):
    """
    CNN feature extractor using ResNet50.
    Extracts local spatial features from face images.
    """

    def __init__(self, pretrained: bool = True, freeze_layers: int = 0):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        cnn = resnet50(weights=weights)

        # Remove final layers
        self.features = nn.Sequential(*list(cnn.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Freeze early layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers: int):
        """Freeze first N layer groups."""
        layer_groups = [
            self.features[:4],  # conv1 + bn1 + relu + maxpool
            self.features[4],  # layer1
            self.features[5],  # layer2
            self.features[6],  # layer3
            self.features[7],  # layer4
        ]

        for i, group in enumerate(layer_groups[:num_layers]):
            for param in group.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Feature tensor [B, 2048, 7, 7]
        """
        features = self.features(x)
        return features


class ViTBranch(nn.Module):
    """
    Vision Transformer branch using DeiT-Small.
    Extracts global contextual features.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Load pretrained DeiT weights if available
        if pretrained and TRANSFORMERS_AVAILABLE:
            self._load_pretrained()

    def _load_pretrained(self):
        """Load pretrained DeiT weights."""
        try:
            from transformers import DeiTForImageClassification

            pretrained_model = DeiTForImageClassification.from_pretrained(
                "facebook/deit-small-patch16-224",
            )

            with torch.no_grad():
                self.patch_embed.weight.copy_(
                    pretrained_model.deit.embeddings.patch_embeddings.projection.weight
                )
                self.patch_embed.bias.copy_(
                    pretrained_model.deit.embeddings.patch_embeddings.projection.bias
                )

            with torch.no_grad():
                self.cls_token.copy_(pretrained_model.deit.embeddings.cls_token)

            logger.info("Loaded pretrained DeiT-Small weights")
        except Exception as e:
            logger.warning(f"Could not load pretrained DeiT weights: {e}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ViT features.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Tuple of (CLS token [B, embed_dim], attention weights)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Store attention weights
        attn_weights = []

        # Transformer blocks
        for block in self.blocks:
            x, attn = block(x)
            attn_weights.append(attn)

        x = self.norm(x)

        # Return CLS token and attention
        return x[:, 0], torch.stack(attn_weights)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop_rate, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention output."""
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class FeatureFusion(nn.Module):
    """Fuses CNN and ViT features using attention-based fusion."""

    def __init__(self, cnn_dim: int = 2048, vit_dim: int = 384, hidden_dim: int = 1024):
        super().__init__()

        # Project CNN features
        self.cnn_project = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Project ViT features
        self.vit_project = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Cross-attention for fusion
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse CNN and ViT features.

        Args:
            cnn_feat: CNN features [B, 2048] (after pooling)
            vit_feat: ViT CLS token [B, 384]

        Returns:
            Fused features [B, hidden_dim]
        """
        # Project features
        cnn_proj = self.cnn_project(cnn_feat)  # [B, hidden_dim]
        vit_proj = self.vit_project(vit_feat)  # [B, hidden_dim]

        # Cross-attention fusion
        cnn_expanded = cnn_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        vit_expanded = vit_proj.unsqueeze(1)  # [B, 1, hidden_dim]

        # Attend CNN features to ViT features
        cnn_attended, _ = self.cross_attn(cnn_expanded, vit_expanded, vit_expanded)

        # Attend ViT features to CNN features
        vit_attended, _ = self.cross_attn(vit_expanded, cnn_expanded, cnn_expanded)

        # Concatenate and fuse
        combined = torch.cat(
            [cnn_attended.squeeze(1), vit_attended.squeeze(1)], dim=1
        )  # [B, hidden_dim * 2]

        fused = self.fusion_mlp(combined)  # [B, hidden_dim]

        return fused


class HCiTModel(nn.Module):
    """
    Hybrid CNN + Vision Transformer Model for Deepfake Detection.

    Architecture:
    1. CNN Branch (ResNet50) - extracts local spatial features
    2. ViT Branch (DeiT-Small) - captures global contextual information
    3. Feature Fusion - combines features using cross-attention
    4. Classification Head - outputs real/fake prediction

    Reference: HCiT - Hybrid CNN and Image Transformer for Deepfake Detection
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        self.config = config or ModelConfig()

        # CNN Branch
        self.cnn = CNNBranch(
            pretrained=self.config.cnn_pretrained,
            freeze_layers=self.config.cnn_freeze_layers,
        )

        # ViT Branch
        self.vit = ViTBranch(
            img_size=224,
            patch_size=16,
            embed_dim=self.config.vit_embed_dim,
            depth=self.config.vit_depth,
            num_heads=self.config.vit_num_heads,
            mlp_ratio=self.config.vit_mlp_ratio,
            drop_rate=self.config.vit_drop_rate,
            attn_drop_rate=self.config.vit_attn_drop_rate,
            pretrained=self.config.vit_pretrained,
        )

        # Feature Fusion
        cnn_dim = 2048  # ResNet50 output dimension
        self.fusion = FeatureFusion(
            cnn_dim=cnn_dim,
            vit_dim=self.config.vit_embed_dim,
            hidden_dim=self.config.fusion_hidden_dim,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.fusion_hidden_dim, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.output_dim, self.config.num_classes),
        )

        # Attention pooling for video-level prediction
        self.attention_pool = nn.MultiheadAttention(
            self.config.fusion_hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.query_vector = nn.Parameter(
            torch.randn(1, 1, self.config.fusion_hidden_dim)
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False, aggregate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W] or [B, T, 3, H, W] for videos
            return_features: Whether to return intermediate features
            aggregate: Whether to aggregate frame-level predictions

        Returns:
            Classification logits or (logits, features) if return_features=True
        """
        if x.dim() == 5:
            # Video input: [B, T, C, H, W]
            batch_size, num_frames = x.shape[:2]
            x = x.view(batch_size * num_frames, *x.shape[2:])
            was_video = True
        else:
            was_video = False

        # CNN features
        cnn_feat = self.cnn(x)  # [B, 2048, 7, 7]
        cnn_feat = torch.nn.functional.adaptive_avg_pool2d(cnn_feat, (1, 1))
        cnn_feat = cnn_feat.flatten(1)  # [B, 2048]

        # ViT features
        vit_cls, vit_attn = self.vit(x)  # [B, 384], [B, num_heads, seq, seq]

        # Fuse features
        fused = self.fusion(cnn_feat, vit_cls)  # [B, fusion_hidden_dim]

        # Classification
        logits = self.classifier(fused)  # [B, num_classes]

        if was_video and aggregate:
            # Aggregate frame-level predictions
            fused = fused.view(batch_size, num_frames, -1)
            query = self.query_vector.expand(batch_size, -1, -1)
            pooled, _ = self.attention_pool(query, fused, fused)
            pooled = pooled.squeeze(1)
            logits = self.classifier(pooled)

        if return_features:
            return logits, {
                "cnn_features": cnn_feat,
                "vit_features": vit_cls,
                "fused_features": fused,
                "vit_attention": vit_attn,
            }

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fused features for embedding or similarity computation."""
        _, features = self.forward(x, return_features=True)
        return features["fused_features"]

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights from ViT branch."""
        _, features = self.forward(x, return_features=True)
        return features["vit_attention"]


def create_hcit_model(
    num_classes: int = 2, pretrained: bool = True, model_path: Optional[Path] = None
) -> HCiTModel:
    """
    Create and initialize HCiT model.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        model_path: Path to saved model weights

    Returns:
        Initialized HCiT model
    """
    config = ModelConfig(
        num_classes=num_classes, cnn_pretrained=pretrained, vit_pretrained=pretrained
    )

    model = HCiTModel(config)

    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


# Alternative: Simple CNN-only model for faster inference
class SimpleCNNClassifier(nn.Module):
    """
    Simplified CNN classifier for faster inference.
    Uses EfficientNet-B0 backbone.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            num_features = 1280
        except:
            # Fallback to ResNet18
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            num_features = 512

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

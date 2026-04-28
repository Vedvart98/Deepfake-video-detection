"""
Temporal aggregation using Bi-LSTM for video-level predictions.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class TemporalAggregator(nn.Module):
    """
    Bi-LSTM temporal aggregator for frame-level features.
    Captures temporal inconsistencies in deepfake videos.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(
        self, 
        frame_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            frame_features: [B, T, input_dim] - features from all frames
            
        Returns:
            video_features: [B, hidden_dim * 2]
            attention_weights: [B, T] (optional)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(frame_features)  # [B, T, hidden_dim * 2]
        
        # Attention-based pooling
        attn_scores = self.attention(lstm_out)  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
        
        # Weighted sum
        video_features = torch.sum(lstm_out * attn_weights, dim=1)  # [B, hidden_dim * 2]
        
        if return_attention:
            return video_features, attn_weights.squeeze(-1)
        return video_features, None


class VideoLevelClassifier(nn.Module):
    """
    Complete video-level classifier with temporal aggregation.
    """
    
    def __init__(
        self,
        frame_feature_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.temporal_agg = TemporalAggregator(
            input_dim=frame_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(
        self, 
        frame_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            frame_features: [B, T, frame_feature_dim]
            
        Returns:
            logits: [B, num_classes]
            attention_weights: [B, T] (optional)
        """
        video_features, attn = self.temporal_agg(frame_features, return_attention)
        logits = self.classifier(video_features)
        
        return logits, attn


def create_temporal_model(
    frame_feature_dim: int = 512,
    hidden_dim: int = 256,
    num_classes: int = 2,
) -> VideoLevelClassifier:
    """Create temporal video classifier."""
    return VideoLevelClassifier(
        frame_feature_dim=frame_feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
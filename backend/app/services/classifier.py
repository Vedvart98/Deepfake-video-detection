"""
Module 4: Classification Layer
Binary classification with video-level aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AggregationMethod(str, Enum):
    AVERAGE = "average"
    MAX = "max"
    VOTING = "voting"
    ATTENTION = "attention"


@dataclass
class FramePrediction:
    frame_idx: int
    timestamp: float
    label: str
    confidence: float
    probabilities: Tuple[float, float]


@dataclass
class VideoPrediction:
    video_id: str
    label: str
    confidence: float
    fake_score: float
    aggregation_method: str
    frame_predictions: List[FramePrediction]
    method_scores: Optional[Dict[str, float]] = None
    attention_weights: Optional[List[float]] = None


class DeepfakeClassifier:
    """
    Classification layer with frame-level and video-level predictions.
    Handles aggregation of frame predictions to video-level decision.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        aggregation_method: AggregationMethod = AggregationMethod.AVERAGE,
        threshold: float = 0.5,
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.aggregation_method = aggregation_method
        self.threshold = threshold

    @torch.no_grad()
    def predict_frame(self, face_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict single face frame.

        Args:
            face_tensor: Preprocessed face tensor [3, 224, 224]

        Returns:
            Tuple of (fake_probability, real_probability)
        """
        input_tensor = face_tensor.unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)
        probs = F.softmax(outputs, dim=1)

        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()

        return fake_prob, real_prob

    @torch.no_grad()
    def predict_batch(self, face_tensors: torch.Tensor) -> List[Tuple[float, float]]:
        """
        Predict batch of face frames.

        Args:
            face_tensors: Batch of face tensors [B, 3, 224, 224]

        Returns:
            List of (fake_prob, real_prob) tuples
        """
        inputs = face_tensors.to(self.device)
        outputs = self.model(inputs)
        probs = F.softmax(outputs, dim=1)

        results = [(p[0].item(), p[1].item()) for p in probs.cpu().numpy()]

        return results

    def aggregate_predictions(
        self,
        frame_predictions: List[FramePrediction],
        method: Optional[AggregationMethod] = None,
    ) -> Tuple[str, float, List[float]]:
        """
        Aggregate frame predictions to video-level decision.

        Args:
            frame_predictions: List of frame predictions
            method: Aggregation method (uses default if None)

        Returns:
            Tuple of (label, confidence, attention_weights)
        """
        method = method or self.aggregation_method

        fake_probs = [fp.probabilities[0] for fp in frame_predictions]
        real_probs = [fp.probabilities[1] for fp in frame_predictions]

        if method == AggregationMethod.AVERAGE:
            fake_score = np.mean(fake_probs)
            attention_weights = fake_probs / (np.sum(fake_probs) + 1e-8)

        elif method == AggregationMethod.MAX:
            fake_score = np.max(fake_probs)
            max_idx = np.argmax(fake_probs)
            attention_weights = [1.0 if i == max_idx else 0.0]

        elif method == AggregationMethod.VOTING:
            labels = [1 if fp.label == "FAKE" else 0 for fp in frame_predictions]
            fake_score = np.mean(labels)
            attention_weights = None

        elif method == AggregationMethod.ATTENTION:
            fake_scores = np.array(fake_probs)
            attention_weights = self._compute_attention(fake_scores)
            fake_score = np.sum(fake_scores * attention_weights)

        else:
            fake_score = np.mean(fake_probs)
            attention_weights = None

        label = "FAKE" if fake_score >= self.threshold else "REAL"
        confidence = max(fake_score, 1 - fake_score)

        return label, confidence, attention_weights

    def _compute_attention(self, scores: np.ndarray) -> List[float]:
        """Compute attention weights from scores using softmax."""
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        attention = F.softmax(scores_tensor, dim=0)
        return attention.numpy().tolist()

    def predict_video(
        self, face_predictions: List[Tuple[int, float, torch.Tensor]], video_id: str
    ) -> VideoPrediction:
        """
        Full video prediction pipeline.

        Args:
            face_predictions: List of (frame_idx, timestamp, face_tensor)
            video_id: Video identifier

        Returns:
            VideoPrediction with all details
        """
        frame_predictions = []

        for frame_idx, timestamp, face_tensor in face_predictions:
            fake_prob, real_prob = self.predict_frame(face_tensor)
            label = "FAKE" if fake_prob >= self.threshold else "REAL"
            confidence = max(fake_prob, real_prob)

            frame_predictions.append(
                FramePrediction(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    label=label,
                    confidence=confidence,
                    probabilities=(fake_prob, real_prob),
                )
            )

        video_label, video_confidence, attention_weights = self.aggregate_predictions(
            frame_predictions
        )

        fake_score = np.mean([fp.probabilities[0] for fp in frame_predictions])

        return VideoPrediction(
            video_id=video_id,
            label=video_label,
            confidence=video_confidence,
            fake_score=fake_score,
            aggregation_method=self.aggregation_method.value,
            frame_predictions=frame_predictions,
            attention_weights=attention_weights,
        )

    def optimize_threshold(
        self, predictions: List[Tuple[float, int]], metric: str = "f1"
    ) -> float:
        """
        Optimize classification threshold based on validation data.

        Args:
            predictions: List of (probability, true_label) tuples
            metric: Metric to optimize ('f1', 'accuracy', 'balanced')

        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

        probs = np.array([p[0] for p in predictions])
        labels = np.array([p[1] for p in predictions])

        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)

            if metric == "f1":
                score = f1_score(labels, preds)
            elif metric == "accuracy":
                score = accuracy_score(labels, preds)
            elif metric == "balanced":
                score = balanced_accuracy_score(labels, preds)
            else:
                score = f1_score(labels, preds)

            scores.append(score)

        optimal_idx = np.argmax(scores)
        return thresholds[optimal_idx]

    def get_confidence_intervals(
        self, frame_predictions: List[FramePrediction], confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals for video prediction.

        Args:
            frame_predictions: Frame-level predictions
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with confidence interval bounds
        """
        fake_probs = np.array([fp.probabilities[0] for fp in frame_predictions])

        mean = np.mean(fake_probs)
        std = np.std(fake_probs)

        from scipy import stats

        t_value = stats.t.ppf((1 + confidence_level) / 2, len(fake_probs) - 1)

        margin = t_value * std / np.sqrt(len(fake_probs))

        return {
            "mean": float(mean),
            "std": float(std),
            "lower_bound": float(mean - margin),
            "upper_bound": float(mean + margin),
            "confidence_level": confidence_level,
        }

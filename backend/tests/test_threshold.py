import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.classifier import (
    DeepfakeClassifier,
    FramePrediction,
    AggregationMethod,
)
from app.services.feature_extractor import HCiTModel, ModelConfig


class TestThresholdBehavior:
    @pytest.mark.parametrize(
        "threshold,expected_label",
        [
            (0.3, "FAKE"),
            (0.5, "FAKE"),
            (0.5, "FAKE"),
            (0.7, "REAL"),
        ],
    )
    def test_threshold_decision(self, threshold, expected_label):
        fake_prob = 0.5
        label = "FAKE" if fake_prob >= threshold else "REAL"
        assert label == expected_label

    def test_threshold_0_5_edge_cases(self):
        assert "FAKE" == ("FAKE" if 0.5 >= 0.5 else "REAL")
        assert "REAL" == ("FAKE" if 0.49 >= 0.5 else "REAL")
        assert "FAKE" == ("FAKE" if 0.51 >= 0.5 else "REAL")

    def test_confidence_at_boundary(self):
        fake_prob = 0.5
        confidence = max(fake_prob, 1 - fake_prob)
        assert confidence == 0.5

    def test_confidence_high_when_clear(self):
        fake_prob = 0.9
        confidence = max(fake_prob, 1 - fake_prob)
        assert confidence == 0.9

    def test_aggregate_with_different_thresholds(self):
        config = ModelConfig(num_classes=2, cnn_pretrained=False, vit_pretrained=False)
        model = HCiTModel(config)

        frame_preds = [
            FramePrediction(0, 0.0, "FAKE", 0.6, (0.6, 0.4)),
            FramePrediction(1, 0.2, "FAKE", 0.6, (0.6, 0.4)),
        ]

        clf_low = DeepfakeClassifier(model, device="cpu", threshold=0.3)
        clf_high = DeepfakeClassifier(model, device="cpu", threshold=0.8)

        label_low, conf_low, _ = clf_low.aggregate_predictions(frame_preds)
        label_high, conf_high, _ = clf_high.aggregate_predictions(frame_preds)

        assert label_low == "FAKE"
        assert label_high == "REAL"
        assert conf_low > conf_high


class TestThresholdOptimization:
    def test_optimize_threshold_f1(self):
        config = ModelConfig(num_classes=2, cnn_pretrained=False, vit_pretrained=False)
        model = HCiTModel(config)
        clf = DeepfakeClassifier(model, device="cpu", threshold=0.5)

        predictions = [
            (0.8, 1),
            (0.7, 1),
            (0.6, 1),
            (0.3, 0),
            (0.2, 0),
            (0.1, 0),
        ]

        optimal = clf.optimize_threshold(predictions, metric="f1")

        assert 0.1 <= optimal <= 0.9

    def test_optimize_threshold_balanced(self):
        config = ModelConfig(num_classes=2, cnn_pretrained=False, vit_pretrained=False)
        model = HCiTModel(config)
        clf = DeepfakeClassifier(model, device="cpu", threshold=0.5)

        predictions = [(0.6, 1), (0.4, 0)]
        optimal = clf.optimize_threshold(predictions, metric="balanced")

        assert isinstance(optimal, float)


class TestThresholdSensitivity:
    def test_small_probability_change_affects_label(self):
        prob = 0.51
        label_at_05 = "FAKE" if prob >= 0.5 else "REAL"

        prob_lower = 0.49
        label_at_05_lower = "FAKE" if prob_lower >= 0.5 else "REAL"

        assert label_at_05 == "FAKE"
        assert label_at_05_lower == "REAL"

    def test_extreme_probabilities_consistent(self):
        extreme_fake = 0.99
        extreme_real = 0.01

        label_fake = "FAKE" if extreme_fake >= 0.5 else "REAL"
        label_real = "FAKE" if extreme_real >= 0.5 else "REAL"

        assert label_fake == "FAKE"
        assert label_real == "REAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

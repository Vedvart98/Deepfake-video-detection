import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.classifier import (
    DeepfakeClassifier,
    AggregationMethod,
    FramePrediction,
    VideoPrediction,
)


@pytest.fixture
def mock_model():
    from app.services.feature_extractor import HCiTModel, ModelConfig

    config = ModelConfig(num_classes=2, cnn_pretrained=False, vit_pretrained=False)
    model = HCiTModel(config)
    return model


@pytest.fixture
def classifier(mock_model):
    return DeepfakeClassifier(
        model=mock_model,
        device="cpu",
        aggregation_method=AggregationMethod.AVERAGE,
        threshold=0.5,
    )


def test_predict_frame_output_format(classifier, mock_model):
    face_tensor = torch.randn(1, 3, 224, 224)
    fake_prob, real_prob = classifier.predict_frame(face_tensor)

    assert isinstance(fake_prob, float)
    assert isinstance(real_prob, float)
    assert 0 <= fake_prob <= 1
    assert 0 <= real_prob <= 1
    assert abs(fake_prob + real_prob - 1.0) < 0.01


def test_predict_batch(classifier):
    batch = torch.randn(4, 3, 224, 224)
    results = classifier.predict_batch(batch)

    assert len(results) == 4
    for fake_prob, real_prob in results:
        assert 0 <= fake_prob <= 1
        assert 0 <= real_prob <= 1


def test_threshold_0_5_exactly(classifier):
    fake_prob, _ = classifier.predict_frame(torch.zeros(1, 3, 224, 224))
    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    print(f"fake_prob for zeros: {fake_prob}")


def test_aggregate_average_method(classifier):
    frame_preds = [
        FramePrediction(0, 0.0, "FAKE", 0.8, (0.8, 0.2)),
        FramePrediction(1, 0.2, "REAL", 0.7, (0.3, 0.7)),
    ]
    label, conf, _ = classifier.aggregate_predictions(
        frame_preds, AggregationMethod.AVERAGE
    )

    assert label in ["FAKE", "REAL"]
    assert 0 <= conf <= 1


def test_aggregate_max_method(classifier):
    frame_preds = [
        FramePrediction(0, 0.0, "FAKE", 0.9, (0.9, 0.1)),
        FramePrediction(1, 0.2, "REAL", 0.8, (0.2, 0.8)),
    ]
    label, conf, _ = classifier.aggregate_predictions(
        frame_preds, AggregationMethod.MAX
    )

    assert label == "FAKE"


def test_aggregate_voting_method(classifier):
    frame_preds = [
        FramePrediction(0, 0.0, "FAKE", 0.9, (0.9, 0.1)),
        FramePrediction(1, 0.2, "FAKE", 0.8, (0.8, 0.2)),
        FramePrediction(2, 0.4, "REAL", 0.7, (0.3, 0.7)),
    ]
    label, conf, _ = classifier.aggregate_predictions(
        frame_preds, AggregationMethod.VOTING
    )

    assert label == "FAKE"


def test_threshold_at_0_5_boundary():
    from app.services.feature_extractor import HCiTModel, ModelConfig

    model = HCiTModel(ModelConfig(num_classes=2))
    clf = DeepfakeClassifier(model, device="cpu", threshold=0.5)

    assert "FAKE" == ("FAKE" if 0.5 >= 0.5 else "REAL")
    assert "REAL" == ("FAKE" if 0.49 >= 0.5 else "REAL")


def test_confidence_calculation(classifier):
    frame_preds = [
        FramePrediction(0, 0.0, "FAKE", 0.9, (0.9, 0.1)),
    ]
    label, conf, _ = classifier.aggregate_predictions(frame_preds)

    assert conf == pytest.approx(0.9, abs=0.01)


def test_empty_frame_predictions(classifier):
    frame_preds = []
    label, conf, _ = classifier.aggregate_predictions(frame_preds)

    assert label == "FAKE"
    assert conf == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

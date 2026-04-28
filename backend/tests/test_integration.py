import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.video_processor import VideoProcessor
from app.services.face_detector import FaceDetector, BoundingBox, FaceDetectionResult
from app.services.feature_extractor import HCiTModel, ModelConfig


class TestVideoProcessor:
    def test_video_processor_init(self):
        vp = VideoProcessor(fps=5.0)
        assert vp.target_fps == 5.0

    def test_video_processor_get_metadata_nonexistent(self):
        vp = VideoProcessor()
        metadata = vp.get_metadata(Path("nonexistent.mp4"))
        assert metadata is None


class TestFaceDetector:
    def test_face_detector_init(self):
        fd = FaceDetector(confidence_threshold=0.5)
        assert fd.confidence_threshold == 0.5

    def test_bounding_box_properties(self):
        box = BoundingBox(x1=10, y1=20, x2=110, y2=120, confidence=0.9)
        assert box.width == 100
        assert box.height == 100
        assert box.area == 10000
        assert box.center == (60, 70)

    def test_bounding_box_to_dict(self):
        box = BoundingBox(x1=10, y1=20, x2=110, y2=120, confidence=0.9)
        d = box.to_dict()
        assert d["x1"] == 10
        assert d["y1"] == 20
        assert d["width"] == 100
        assert d["confidence"] == 0.9

    def test_face_detection_result_empty(self):
        result = FaceDetectionResult(frame_idx=0, timestamp=0.0)
        assert result.num_faces == 0
        assert not result.has_faces

    def test_get_best_face(self):
        faces = [
            BoundingBox(x1=10, y1=10, x2=50, y2=50, confidence=0.8),
            BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.7),
        ]
        result = FaceDetectionResult(frame_idx=0, timestamp=0.0, faces=faces)
        best = result.get_best_face()
        assert best is not None

    def test_preprocess_for_model(self):
        fd = FaceDetector()
        face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = fd.preprocess_for_model(face)
        assert processed.shape == (3, 224, 224)


class TestFeatureExtractor:
    def test_hcit_model_init(self):
        config = ModelConfig(num_classes=2)
        model = HCiTModel(config)
        assert model is not None

    def test_hcit_forward_shape(self):
        config = ModelConfig(num_classes=2)
        model = HCiTModel(config)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 2)

    def test_cnn_branch_output_shape(self):
        from app.services.feature_extractor import CNNBranch

        cnn = CNNBranch(pretrained=False)
        cnn.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = cnn(x)

        assert out.shape[0] == 1
        assert out.shape[1] == 2048


class TestIntegration:
    def test_end_to_end_prediction_pipeline(self):
        config = ModelConfig(num_classes=2, cnn_pretrained=False, vit_pretrained=False)
        model = HCiTModel(config)
        model.eval()

        from app.services.classifier import DeepfakeClassifier, FramePrediction

        classifier = DeepfakeClassifier(model=model, device="cpu", threshold=0.5)

        face_tensors = [torch.randn(1, 3, 224, 224) for _ in range(5)]
        predictions = []

        for i, tensor in enumerate(face_tensors):
            fake_prob, real_prob = classifier.predict_frame(tensor)
            label = "FAKE" if fake_prob >= 0.5 else "REAL"
            predictions.append(
                FramePrediction(
                    frame_idx=i,
                    timestamp=i * 0.2,
                    label=label,
                    confidence=max(fake_prob, real_prob),
                    probabilities=(fake_prob, real_prob),
                )
            )

        video_label, video_conf, _ = classifier.aggregate_predictions(predictions)

        assert video_label in ["FAKE", "REAL"]
        assert 0 <= video_conf <= 1
        print(f"Video label: {video_label}, confidence: {video_conf:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

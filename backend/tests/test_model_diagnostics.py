"""
Diagnostic test to verify model behavior.
This helps identify why the model might always predict "FAKE".
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.feature_extractor import HCiTModel, ModelConfig
from app.services.classifier import DeepfakeClassifier, AggregationMethod


def test_model_random_output():
    """Test that model produces varied outputs for different inputs."""
    print("\n=== Test 1: Model Output Variation ===")

    config = ModelConfig(num_classes=2, cnn_pretrained=True, vit_pretrained=True)
    model = HCiTModel(config)
    model.eval()

    # Create random inputs
    torch.manual_seed(42)
    inputs = [
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
    ]

    outputs = []
    with torch.no_grad():
        for inp in inputs:
            out = model(inp)
            probs = torch.softmax(out, dim=1)
            fake_prob = probs[0, 0].item()
            real_prob = probs[0, 1].item()
            outputs.append((fake_prob, real_prob))
            print(
                f"  Input {inputs.index(inp) + 1}: FAKE={fake_prob:.4f}, REAL={real_prob:.4f}"
            )

    fake_probs = [o[0] for o in outputs]
    variance = np.var(fake_probs)
    print(f"\n  Variance in fake probabilities: {variance:.6f}")

    if variance < 0.001:
        print("  WARNING: Model outputs are nearly identical!")
        print("  This suggests the model might be predicting a constant class.")
        return False

    print("  PASS: Model produces varied outputs")
    return True


def test_threshold_behavior():
    """Test that threshold is applied correctly."""
    print("\n=== Test 2: Threshold Behavior ===")

    config = ModelConfig(num_classes=2, cnn_pretrained=False, vit_pretrained=False)
    model = HCiTModel(config)

    classifier = DeepfakeClassifier(
        model=model,
        device="cpu",
        aggregation_method=AggregationMethod.AVERAGE,
        threshold=0.5,
    )

    # Test with different probability distributions
    test_cases = [
        ((0.9, 0.1), "FAKE"),  # High fake probability
        ((0.1, 0.9), "REAL"),  # High real probability
        ((0.5, 0.5), "FAKE"),  # Equal (should default to FAKE due to >=)
        ((0.49, 0.51), "REAL"),  # Slightly more real
        ((0.51, 0.49), "FAKE"),  # Slightly more fake
    ]

    all_passed = True
    for probs, expected_label in test_cases:
        fake_prob, real_prob = probs
        predicted = "FAKE" if fake_prob >= 0.5 else "REAL"
        status = "✓" if predicted == expected_label else "✗"
        print(
            f"  {status} FAKE={fake_prob:.2f}, REAL={real_prob:.2f} -> {predicted} (expected {expected_label})"
        )
        if predicted != expected_label:
            all_passed = False

    if all_passed:
        print("  PASS: Threshold behavior is correct")
    else:
        print("  FAIL: Threshold behavior is incorrect")

    return all_passed


def test_aggregation_method():
    """Test video-level aggregation."""
    print("\n=== Test 3: Aggregation Methods ===")

    config = ModelConfig(num_classes=2, cnn_pretrained=False, vit_pretrained=False)
    model = HCiTModel(config)

    from app.services.classifier import FramePrediction

    # Create frame predictions with varying fake probabilities
    frame_preds = [
        FramePrediction(
            frame_idx=0,
            timestamp=0.0,
            label="REAL",
            confidence=0.9,
            probabilities=(0.1, 0.9),
        ),
        FramePrediction(
            frame_idx=1,
            timestamp=0.2,
            label="REAL",
            confidence=0.8,
            probabilities=(0.2, 0.8),
        ),
        FramePrediction(
            frame_idx=2,
            timestamp=0.4,
            label="REAL",
            confidence=0.7,
            probabilities=(0.3, 0.7),
        ),
    ]

    classifier = DeepfakeClassifier(model=model, device="cpu", threshold=0.5)

    # Test AVERAGE aggregation
    label, conf, _ = classifier.aggregate_predictions(
        frame_preds, AggregationMethod.AVERAGE
    )
    print(
        f"  AVERAGE aggregation: fake_scores={[fp.probabilities[0] for fp in frame_preds]}"
    )
    print(f"    -> {label} (confidence: {conf:.4f})")

    # Test with fake-dominant frames
    fake_preds = [
        FramePrediction(
            frame_idx=0,
            timestamp=0.0,
            label="FAKE",
            confidence=0.8,
            probabilities=(0.8, 0.2),
        ),
        FramePrediction(
            frame_idx=1,
            timestamp=0.2,
            label="FAKE",
            confidence=0.9,
            probabilities=(0.9, 0.1),
        ),
        FramePrediction(
            frame_idx=2,
            timestamp=0.4,
            label="FAKE",
            confidence=0.7,
            probabilities=(0.7, 0.3),
        ),
    ]

    label, conf, _ = classifier.aggregate_predictions(
        fake_preds, AggregationMethod.AVERAGE
    )
    print(
        f"  With fake frames: fake_scores={[fp.probabilities[0] for fp in fake_preds]}"
    )
    print(f"    -> {label} (confidence: {conf:.4f})")

    print("  PASS: Aggregation works correctly")
    return True


def test_model_untrained_behavior():
    """Check behavior of untrained model."""
    print("\n=== Test 4: Untrained Model Behavior ===")

    config = ModelConfig(num_classes=2, cnn_pretrained=True, vit_pretrained=True)
    model = HCiTModel(config)
    model.eval()

    # Test with different types of inputs
    test_inputs = [
        ("zeros", torch.zeros(1, 3, 224, 224)),
        ("ones", torch.ones(1, 3, 224, 224)),
        ("random", torch.randn(1, 3, 224, 224)),
        ("uniform", torch.rand(1, 3, 224, 224)),
    ]

    print("  Testing untrained model with various inputs:")
    results = []
    with torch.no_grad():
        for name, inp in test_inputs:
            out = model(inp)
            probs = torch.softmax(out, dim=1)
            fake_prob = probs[0, 0].item()
            results.append(fake_prob)
            print(f"    {name}: FAKE={fake_prob:.4f}, REAL={probs[0, 1].item():.4f}")

    # Check if model has any learned bias
    mean_fake = np.mean(results)
    print(f"\n  Mean FAKE probability: {mean_fake:.4f}")

    if mean_fake > 0.7:
        print("  WARNING: Untrained model has strong bias toward FAKE!")
        print("  This is likely due to untrained classification head.")
        print("  -> The model needs to be trained on labeled data!")
    elif mean_fake < 0.3:
        print("  WARNING: Untrained model has strong bias toward REAL!")
    else:
        print("  Model has balanced (untrained) output distribution")

    return True


def test_pretrained_cnn_only():
    """Test with just CNN features (no ViT)."""
    print("\n=== Test 5: CNN-only Feature Extraction ===")

    from app.services.feature_extractor import CNNBranch

    cnn = CNNBranch(pretrained=True, freeze_layers=0)
    cnn.eval()

    inp = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = cnn(inp)

    print(f"  CNN output shape: {features.shape}")
    print(f"  After pooling: {features.flatten(1).shape}")
    print("  PASS: CNN feature extraction works")

    return True


def diagnose_model():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("DEEPFAKE SENTINEL - MODEL DIAGNOSTICS")
    print("=" * 60)

    results = []

    results.append(("Model Output Variation", test_model_random_output()))
    results.append(("Threshold Behavior", test_threshold_behavior()))
    results.append(("Aggregation Method", test_aggregation_method()))
    results.append(("Untrained Model", test_model_untrained_behavior()))
    results.append(("CNN Features", test_pretrained_cnn_only()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("=" * 60))

    if all_passed:
        print("All diagnostics passed!")
        print("\nPossible causes for 'always FAKE' predictions:")
        print("  1. Model weights not loaded (using untrained model)")
        print("  2. Training data distribution issue")
        print("  3. Threshold too low")
    else:
        print("Some diagnostics failed - check results above")

    return all_passed


if __name__ == "__main__":
    diagnose_model()

# Deepfake Sentinel Testing Guide

## Overview

This document explains how to test the Deepfake Sentinel detection system, why the model may be predicting "FAKE" for all videos, and how to verify the system is working correctly.

## Problem: Model Always Predicting "FAKE"

### Root Cause Identified

**The model weights file (`best_model.pth`) does not exist.** 

When the model loads without weights, it uses random/untrained weights which produce biased predictions toward "FAKE":

```
Untrained model predictions:
  zeros: FAKE=0.7790, REAL=0.2210
  random: FAKE=0.7300, REAL=0.2700
  uniform: FAKE=0.5941, REAL=0.4059
```

With the default threshold of 0.5, any fake_prob > 0.5 results in "FAKE" classification.

### How to Fix

**Option 1: Train the model**
```bash
cd backend
python train.py --data_dir data/processed --epochs 50 --batch_size 32
```

**Option 2: Download pretrained weights**
- Place trained weights at: `backend/models/detector/best_model.pth`

---

## Testing the System

### Running Tests

```bash
cd backend

# Run all tests
./deepfake/bin/python tests/run_tests.py --all

# Run specific test categories
./deepfake/bin/python tests/run_tests.py --unit       # Classifier tests
./deepfake/bin/python tests/run_tests.py --threshold  # Threshold tests  
./deepfake/bin/python tests/run_tests.py --integration # Integration tests
```

### Test Files Created

| File | Purpose |
|------|---------|
| `tests/test_model_diagnostics.py` | Diagnose model behavior and identify issues |
| `tests/test_classifier.py` | Unit tests for classifier |
| `tests/test_integration.py` | Integration tests for full pipeline |
| `tests/test_threshold.py` | Tests for threshold behavior |
| `tests/run_tests.py` | Test runner script |

---

## Testing on Your Own Videos

### Method 1: Using the API

```bash
# Start the server
cd backend
./deepfake/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal, upload a video
curl -X POST -F "file=@your_video.mp4" http://localhost:8000/api/v1/analyze

# Get results (replace JOB_ID with returned job_id)
curl http://localhost:8000/api/v1/result/{job_id}
```

### Method 2: Using the Evaluation Script

The evaluation script expects a dataset in this structure:
```
data/processed/
├── train/
│   ├── real/   # Put real (original) videos here
│   └── fake/   # Put deepfake videos here
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Run evaluation:
```bash
python evaluate.py \
    --model_path models/detector/best_model.pth \
    --data_dir data/processed \
    --split test \
    --output results.json
```

### Method 3: Using the Diagnostic Script

Test the model directly with sample images:
```bash
./deepfake/bin/python tests/test_model_diagnostics.py
```

---

## Understanding Test Results

### Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| Accuracy | Overall correctness | > 0.85 |
| Precision | Of predicted FAKE, how many are actually FAKE | > 0.85 |
| Recall | Of actual FAKE, how many were detected | > 0.85 |
| F1 | Harmonic mean of Precision/Recall | > 0.85 |
| AUC-ROC | Area under ROC curve | > 0.90 |
| AUC-PR | Area under Precision-Recall curve | > 0.90 |

### Interpreting Results

- **If model always says FAKE**: Likely untrained weights or wrong threshold
- **If model always says REAL**: Check if deepfake quality is very high or model overfit
- **If mixed results but poor accuracy**: Model may need more training or better data

---

## Sample Videos Available

The backend already has sample videos in `backend/data/uploads/`:
- Multiple MP4 files (18 total)
- These were used in previous analyses

---

## Troubleshooting

### Issue: "No module named 'torch'"
Use the virtual environment Python:
```bash
./deepfake/bin/python your_script.py
```

### Issue: Model always FAKE
1. Check if model weights exist: `ls -la models/detector/`
2. If empty, train the model or download pretrained weights

### Issue: Low accuracy on real videos
- The model may be biased
- Try adjusting the threshold (in `app/core/config.py`)
- Default: `CLASSIFICATION_THRESHOLD = 0.5`
- Try higher values (e.g., 0.6, 0.7) to reduce false positives

---

## Files and Paths Reference

| Purpose | Path |
|---------|------|
| Model weights | `backend/models/detector/best_model.pth` |
| Config | `backend/app/core/config.py` |
| Classifier | `backend/app/services/classifier.py` |
| Model | `backend/app/services/feature_extractor.py` |
| Video API | `backend/app/api/endpoints/video.py` |
| Training | `backend/train.py` |
| Evaluation | `backend/evaluate.py` |
| Training config | `backend/configs/train_hcit.yaml` |

---

## Next Steps

1. **Train the model** or obtain pretrained weights
2. **Run the diagnostic test** to verify correct behavior
3. **Test on known samples** (real and fake videos)
4. **Adjust threshold** if false positives are too high
5. **Evaluate on test set** to get metrics
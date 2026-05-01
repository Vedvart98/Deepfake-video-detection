# DeepFake Sentinel - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [How It Works: Input to Output](#how-it-works-input-to-output)
3. [Models in This Project](#models-in-this-project)
4. [Training Process](#training-process)
5. [Dataset Information](#dataset-information)
6. [Confusion Matrix & Output Interpretation](#confusion-matrix--output-interpretation)

---

## 1. Project Overview

**DeepFake Sentinel** is a comprehensive deepfake video detection system that uses a hybrid CNN + Vision Transformer architecture (called HCiT - Hybrid CNN and Image Transformer) combined with multi-modal analysis (visual + audio-visual consistency) and explainable AI (Grad-CAM visualizations).

**Tech Stack:**
| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python 3.11 |
| Deep Learning | PyTorch 2.x |
| Face Detection | YOLOv8-Face |
| Feature Extraction | ResNet50 + DeiT-Small |
| Audio Processing | librosa, Wav2Vec2 |
| Grad-CAM | pytorch-grad-cam |
| Frontend | React 18 + Vite + TypeScript |
| Containerization | Docker |

---

## 2. How It Works: Input to Output

The system processes videos through **7 main modules** in a pipeline:

### Module 1: Video Input Processing
- **Input**: Video file (MP4, AVI, MOV, WebM)
- **Process**:
  1. Accepts video upload via API
  2. Extracts metadata (duration, fps, resolution, codec)
  3. Extracts frames at configurable FPS (default: 5 FPS)
  4. Detects scene changes using frame differencing
  5. Selects key frames for processing
- **Output**: List of video frames with timestamps

**Code Location**: `backend/app/services/video_processor.py`

### Module 2: Face Detection (YOLOv8)
- **Input**: Video frames from Module 1
- **Process**:
  1. Passes each frame through YOLOv8 face detector
  2. Detects all faces with bounding boxes and 5-point landmarks
  3. Applies Non-Maximum Suppression (NMS) with IoU threshold 0.45
  4. Filters faces by confidence threshold (default: 0.5)
  5. Adds margin around face bounding boxes (20%)
  6. Assesses face quality based on size and confidence
  7. Aligns faces using eye landmarks
- **Output**: FaceDetectionResult with detected faces (bounding boxes + landmarks)

**Code Location**: `backend/app/services/face_detector.py`

### Module 3: Feature Extraction (HCiT - Hybrid CNN + Vision Transformer)
- **Input**: Face crops (224x224)
- **Process**:
  1. **CNN Branch (ResNet50)**:
     - Extracts local spatial features from face images
     - Output: 2048-dimensional feature maps (7x7 spatial)
     - Applies adaptive average pooling to get 2048-dim vector
  
  2. **ViT Branch (DeiT-Small)**:
     - Divides image into 16x16 patches
     - Uses patch embedding to convert to 384-dim tokens
     - Adds class token (CLS) for classification
     - Passes through 12 transformer blocks with 6 attention heads
     - Extracts global contextual features from CLS token
     - Output: 384-dimensional CLS token + attention weights
  
  3. **Feature Fusion**:
     - Projects CNN features to 1024-dim
     - Projects ViT features to 1024-dim
     - Applies cross-attention to fuse features
     - Concatenates and passes through fusion MLP
     - Output: 1024-dimensional fused features
- **Output**: Fused feature vectors for each face

**Code Location**: `backend/app/services/feature_extractor.py`

### Module 4: Classification Layer
- **Input**: Fused features from Module 3
- **Process**:
  1. Passes fused features through classification head:
     - Linear(1024 → 512) + LayerNorm + GELU + Dropout
     - Linear(512 → 2) for binary classification (FAKE/REAL)
  
  2. **Frame-level prediction**:
     - Applies softmax to get probabilities
     - Returns (fake_probability, real_probability)
     - Compares against threshold (default: 0.5)
  
  3. **Video-level aggregation** (multiple methods):
     - **Average**: Mean of all frame fake probabilities
     - **Max**: Maximum fake probability across frames
     - **Voting**: Percentage of frames classified as FAKE
     - **Attention**: Weighted average using learned attention
  
  4. **Threshold optimization**:
     - Tests thresholds from 0.1 to 0.9
     - Optimizes for F1-score, accuracy, or balanced accuracy
- **Output**: Frame predictions + Video prediction with confidence

**Code Location**: `backend/app/services/classifier.py`

### Module 5: Grad-CAM Explainability
- **Input**: Face tensor + model
- **Process**:
  1. Selects target layer (ViT's last transformer block or CNN's layer4)
  2. Computes gradients of target class score w.r.t. feature maps
  3. Applies global average pooling to weights
  4. Reweights feature maps and applies ReLU
  5. Upsamples to input resolution
  6. Generates heatmap with values in [0, 1]
  7. Applies colormap (JET) for visualization
  8. Overlays heatmap on original frame with alpha blending
  9. Identifies manipulation regions using contour detection
- **Output**: Heatmap visualization showing which regions contributed most to the prediction

**Code Location**: `backend/app/services/gradcam.py`

### Module 6: Audio-Visual Consistency
- **Input**: Video file + face frames
- **Process**:
  1. **Audio Extraction**: Uses librosa to extract audio at 16kHz mono
  
  2. **Speech Feature Extraction**:
     - Primary: Wav2Vec2 (facebook/wav2vec2-base-960h)
     - Fallback: MFCC + delta + delta-delta features (39 dimensions)
  
  3. **Visual Feature Extraction**:
     - Extracts mouth region (65%-85% vertical, 25%-75% horizontal)
     - Resizes to 64x32
     - Flattens to 2048-dimensional vector
  
  4. **Sync Score Computation**:
     - Computes cross-correlation between audio and visual features
     - Normalizes by length
     - Returns sync score in [0, 1]
  
  5. **Desynchronization Detection**:
     - Compares sync score against threshold (0.7)
     - Returns is_desynced flag
- **Output**: AudioVisualResult with sync_score, is_desynced, confidence, temporal_alignment

**Code Location**: `backend/app/services/audio_visual.py`

### Module 7: Web Interface
- **Input**: User interactions
- **Process**:
  1. Drag-and-drop video upload
  2. Real-time progress updates via polling
  3. Results display with confidence scores
  4. Heatmap overlay playback
  5. Batch processing support
- **Output**: User-friendly UI for video upload and results visualization

**Code Location**: `frontend/src/`

---

## 3. Models in This Project

### Model 1: YOLOv8 (Face Detection)
- **Purpose**: Detect faces in video frames
- **Architecture**: YOLOv8n (nano variant) - 3.2M parameters
- **Input**: RGB image (any resolution)
- **Output**: Bounding boxes + confidence scores + 5-point facial landmarks
- **Configuration**:
  - Confidence threshold: 0.5
  - IoU threshold: 0.45
  - Image size: 640x640

### Model 2: HCiT (Hybrid CNN + Vision Transformer)
- **Purpose**: Extract features and classify faces as REAL or FAKE

#### 2a. CNN Branch (ResNet50)
- **Purpose**: Extract local spatial features
- **Input**: 224x224 RGB face image
- **Output**: 2048-dimensional feature vector (after pooling)
- **Pretrained**: ImageNet1K V2

#### 2b. ViT Branch (DeiT-Small)
- **Purpose**: Extract global contextual features
- **Input**: 224x224 RGB face image
- **Output**: 384-dimensional CLS token + attention weights
- **Configuration**:
  - Patch size: 16x16
  - Embedding dim: 384
  - Depth: 12 blocks
  - Attention heads: 6
  - MLP ratio: 4.0

#### 2c. Feature Fusion
- **Purpose**: Combine CNN and ViT features
- **Method**: Cross-attention based fusion
- **Output**: 1024-dimensional fused features

#### 2d. Classification Head
- **Purpose**: Binary classification (REAL/FAKE)
- **Output**: 2-class logits (cross-entropy loss)

### Model 3: Wav2Vec2 (Audio Processing)
- **Purpose**: Extract speech embeddings for audio-visual consistency
- **Model**: facebook/wav2vec2-base-960h
- **Input**: Raw audio waveform (16kHz)
- **Output**: 768-dimensional hidden states (6th layer)

---

## 4. Training Process

### Training Configuration (from `configs/train_hcit.yaml`)

```yaml
model:
  name: "HCiT-ResNet50-DeiT-S"
  num_classes: 2
  pretrained: true

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.00001
  scheduler: "cosine"
  warmup_epochs: 5

optimizer:
  type: "AdamW"
  betas: [0.9, 0.999]

augmentation:
  random_horizontal_flip: 0.5
  random_rotation: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
  gaussian_blur: 0.1
  mixup_alpha: 0.2
  cutmix_alpha: 0.2
```

### Training Process

1. **Data Loading**:
   - Dataset class: `DeepfakeDataset` in `train.py`
   - Expects directory structure: `data/{split}/real/` and `data/{split}/fake/`
   - Supports JPG and PNG images

2. **Preprocessing**:
   - Resize to 224x224
   - Convert BGR to RGB
   - Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Convert to CHW format

3. **Training Loop**:
   ```python
   for epoch in range(epochs):
       # Training
       model.train()
       for batch in train_loader:
           images, labels = batch
           optimizer.zero_grad()
           with autocast():  # Mixed precision
               outputs = model(images)
               loss = criterion(outputs, labels)
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()
       
       # Validation
       model.eval()
       with torch.no_grad():
           for batch in val_loader:
               # Compute metrics: accuracy, F1, precision, recall
   ```

4. **Optimization**:
   - Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
   - Scheduler: CosineAnnealingLR
   - Mixed precision training with GradScaler

5. **Model Checkpointing**:
   - Saves best model based on validation accuracy
   - Output: `models/detector/best_model.pth`

### Training Command

```bash
cd backend
python train.py --config configs/train_hcit.yaml --epochs 50 --batch_size 32
```

---

## 5. Dataset Information

### Datasets Used

| Dataset | Description |
|---------|-------------|
| **FaceForensics++ (FF++)** | 1000 original videos + synthesized deepfakes using various methods (Face2Face, FaceSwap, NeuralTextures) |
| **Celeb-DF v2** | High-quality deepfakes of celebrities from YouTube |
| **WildDeepfake** | Real-world deepfakes collected from the internet |

### Data Directory Structure

```
data/processed/
├── train/
│   ├── real/          # Real face images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── fake/          # Fake/deepfake face images
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Data Split
- Training: 80%
- Validation: 10%
- Test: 10%

---

## 6. Confusion Matrix & Output Interpretation

### Evaluation Metrics

When you run the evaluation script (`evaluate.py`), it computes:

```python
metrics = {
    "accuracy": accuracy_score(labels, preds),
    "precision": precision_score(labels, pred),
    "recall": recall_score(labels, preds),
    "f1": f1_score(labels, preds),
    "auc_roc": roc_auc_score(labels, probs),
    "auc_pr": average_precision_score(labels, probs),
    "confusion_matrix": confusion_matrix(labels, preds),
}
```

### Confusion Matrix

For binary classification (REAL vs FAKE):

```
                    Predicted
                  REAL    FAKE
Actual  REAL     [TN]    [FP]
        FAKE     [FN]    [TP]
```

Where:
- **TN (True Negative)**: Real video correctly identified as REAL
- **TP (True Positive)**: Fake video correctly identified as FAKE
- **FP (False Positive)**: Real video incorrectly identified as FAKE
- **FN (False Negative)**: Fake video incorrectly identified as REAL

### How to Interpret the Output

#### 1. Run Evaluation
```bash
python evaluate.py --model_path models/detector/best_model.pth --data_dir data/processed --split test
```

#### 2. Output Example
```
=== Evaluation Results ===
Accuracy: 0.9234
Precision: 0.9156
Recall: 0.9312
F1-Score: 0.9233
AUC-ROC: 0.9678
AUC-PR: 0.9512
Confusion Matrix:
[[450  50]
 [ 35 465]]
```

#### 3. Interpretation

| Metric | Value | Meaning |
|--------|-------|---------|
| **Accuracy** | 92.34% | Overall, 92.34% of predictions are correct |
| **Precision** | 91.56% | Of videos predicted as FAKE, 91.56% are actually FAKE |
| **Recall** | 93.12% | Of actual FAKE videos, 93.12% are correctly detected |
| **F1-Score** | 92.33% | Harmonic mean of precision and recall |
| **AUC-ROC** | 96.78% | Excellent discrimination between REAL and FAKE |
| **AUC-PR** | 95.12% | High precision-recall tradeoff |

**Confusion Matrix Interpretation**:
```
              Predicted
            REAL    FAKE
Actual REAL  450     50   → 450 real videos correctly identified, 50 falsely accused
       FAKE   35    465   → 465 fakes caught, 35 slipped through
```

- **Total samples**: 1000
- **Correct predictions**: 450 + 465 = 915
- **Incorrect predictions**: 50 + 35 = 85
- **Accuracy**: 915/1000 = 91.5%

### What the API Returns

When you analyze a video via the API:

```json
{
  "job_id": "uuid",
  "status": "COMPLETED",
  "video_prediction": {
    "video_id": "uuid",
    "label": "FAKE",           // or "REAL"
    "confidence": 0.87,         // 0.0 to 1.0
    "fake_score": 0.87,        // Average fake probability
    "aggregation_method": "average",
    "frame_predictions": [
      {
        "frame_idx": 0,
        "timestamp": 0.0,
        "label": "FAKE",
        "confidence": 0.92,
        "face_boxes": [{"x1": 100, "y1": 50, "x2": 200, "y2": 180, "confidence": 0.95}]
      },
      ...
    ]
  },
  "audio_visual": {
    "sync_score": 0.65,
    "is_desynced": true,
    "confidence": 0.70,
    "temporal_alignment": [...]
  }
}
```

### Output Fields Explained

| Field | Description |
|-------|-------------|
| `label` | "FAKE" if fake_score ≥ 0.5, else "REAL" |
| `confidence` | Max(fake_score, 1-fake_score) - how confident the model is |
| `fake_score` | Average probability of being FAKE across all frames |
| `frame_predictions` | Per-frame predictions with face bounding boxes |
| `audio_visual.sync_score` | Audio-visual consistency score (0-1) |
| `audio_visual.is_desynced` | True if sync_score < 0.7 (potential lip-sync fake) |

---

## Quick Start

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Train Model
```bash
cd backend
python train.py --config configs/train_hcit.yaml --epochs 50 --batch_size 32
```

### Evaluate Model
```bash
python evaluate.py --model_path models/detector/best_model.pth --data_dir data/processed
```

---

## References

- HCiT: Hybrid CNN and Image Transformer for Deepfake Detection
- FaceForensics++ Dataset
- Celeb-DF Dataset
- Wav2Vec2: Self-supervised learning of speech features
- pytorch-grad-cam for explainable AI
- Ultralytics YOLOv8
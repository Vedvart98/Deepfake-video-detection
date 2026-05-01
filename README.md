# DeepFake Sentinel

<<<<<<< HEAD
A comprehensive deepfake video detection system using **HCiT (Hybrid CNN + Vision Transformer)** architecture with multi-modal analysis (visual + audio-visual consistency), **Grad-CAM explainability**, and production-ready deployment.
=======
A deepfake video detection system that uses an EfficientNet-B4 model trained via fastai, with YOLOv8 face detection and temporal aggregation for robust video-level predictions.
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DEEPFAKE SENTINEL                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Module 1: Video Input Processing                                   │
│      │                                                              │
│      ▼                                                              │
│  Module 2: Face Detection (YOLOv8)                                   │
│      │                                                              │
│      ▼                                                              │
<<<<<<< HEAD
│  Module 3: Feature Extraction (HCiT - CNN + ViT)                     │
│      │                                                              │
│      ├──────────────────────────────────────┐                       │
│      ▼                                      ▼                       │
│  Module 4: Classification          Module 5: Grad-CAM               │
│      │                                      │                       │
│      └──────────────────────────────────────┼───────────────────┐     │
│                                             │                   │     │
│                                             ▼                   │     │
│                                     Module 6: Audio-Visual  │     │
│                                         Consistency          │     │
│                                             │               │     │
│                                             └───────────────┼─┘     │
│                                                             │       │
│                                                             ▼       │
│                                                     Module 7: Web UI  │
=======
│  Module 3: Frame Classification (EfficientNet-B4)                   │
│      │                                                              │
│      ▼                                                              │
│  Module 4: Temporal Aggregation & Video-Level Prediction            │
│      │                                                              │
│      ▼                                                              │
│  Module 5: API & Web Interface                                      │
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

<<<<<<< HEAD
- **HCiT Architecture**: Hybrid CNN (ResNet50) + Vision Transformer (DeiT-Small) with Cross-Attention Fusion
- **YOLOv8 Face Detection**: High-accuracy face detection with quality assessment
- **Grad-CAM Explainability**: Visual heatmaps showing manipulated regions
- **Audio-Visual Consistency**: Lip-sync analysis using Wav2Vec2 and cross-correlation
- **Multi-Frame Aggregation**: Attention-based video-level prediction
- **Modern Tech Stack**: FastAPI + React 18 + TypeScript + Tailwind CSS
=======
- **EfficientNet-B4 Model**: Trained on Kaggle using fastai with MixUp, CutMix, and mixed precision (FP16)
- **YOLOv8 Face Detection**: High-accuracy face detection and cropping from video frames
- **Temporal Aggregation**: Multiple aggregation methods (average, max, voting, attention) for video-level predictions
- **Progressive Training**: Freeze/unfreeze strategy with differential learning rates and early stopping
- **FastAPI Backend**: Async REST API with job tracking and WebSocket support
- **Modern Frontend**: React 18 + Vite + TypeScript with Tailwind CSS
- **Model Compatibility**: fastai `load_learner` export format (`.pkl`)
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python 3.11 |
<<<<<<< HEAD
| Deep Learning | PyTorch 2.x |
| Primary Model | HCiT (ResNet50 + DeiT-Small) |
| Face Detection | YOLOv8-Face |
| Audio Processing | librosa, Wav2Vec2 |
| Explainability | Grad-CAM, HiResCAM, AblationCAM |
=======
| Deep Learning | fastai 2.x / PyTorch 2.x |
| Model Architecture | EfficientNet-B4 |
| Face Detection | YOLOv8 (Ultralytics) |
| Training Platform | Kaggle Notebooks |
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
| Frontend | React 18 + Vite + TypeScript |
| Styling | Tailwind CSS |
| Containerization | Docker |

## Project Structure

```
deepfake-sentinel/
├── backend/                 # Python FastAPI Backend
│   ├── app/
<<<<<<< HEAD
│   │   ├── api/            # API endpoints (video analysis)
│   │   ├── core/          # Configuration
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # Business logic (Modules 1-6)
│   │   │   ├── video_processor.py   # Module 1: Video input
│   │   │   ├── face_detector.py     # Module 2: YOLOv8 detection
│   │   │   ├── feature_extractor.py # Module 3: HCiT model
│   │   │   ├── gradcam.py           # Module 5: Explainability
│   │   │   └── audio_visual.py     # Module 6: Audio-Visual
│   │   └── ml/             # Model management
│   │       └── model_manager.py  # HCiT + EfficientNet support
│   ├── models/             # Trained model weights
│   │   ├── hcit_model.pth       # HCiT trained weights (primary)
│   │   └── deepfake_model-3.pkl # Legacy EfficientNet (fallback)
│   ├── configs/            # Training configurations
│   │   └── train_hcit.yaml
│   └── train_hcit.py       # HCiT training script
├── frontend/               # React + Vite Frontend
│   ├── src/
│   │   ├── components/     # Upload, Results, Dashboard
│   │   ├── hooks/          # Custom React hooks
│   │   └── services/       # API client
│   └── package.json
└── README.md
=======
│   │   ├── api/            # API endpoints (video analysis, health, model info)
│   │   ├── core/          # Configuration and settings
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   │   ├── face_detector.py    # YOLOv8 face detection
│   │   │   ├── classifier.py        # Frame/video classification & aggregation
│   │   │   ├── video_processor.py  # Video frame extraction
│   │   │   └── gradcam.py          # Explainability (Grad-CAM)
│   │   └── ml/
│   │       ├── model_manager.py  # fastai model loading & inference
│   │       └── inference.py       # ML utilities
│   ├── models/             # Trained model weights (deepfake_model-3.pkl)
│   └── tests/             # Backend tests
├── frontend/               # React + Vite Frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── hooks/          # Custom React hooks
│   │   └── services/       # API client
│   └── package.json
├── notebooks/              # Training & experimentation notebooks
│   └── deepfake.ipynb    # Kaggle training notebook (EfficientNet-B4)
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Training Notebook

The model was trained using the Kaggle notebook located at `notebooks/deepfake.ipynb`.

### Notebook Details

| Attribute | Value |
|-----------|-------|
| Model Architecture | EfficientNet-B4 |
| Input Image Size | 380×380 (optimal for B4) |
| Framework | fastai 2.x |
| Mixed Precision | FP16 (`to_fp16()`) |
| Augmentation | MixUp, CutMix, affine transforms, lighting |
| Training Strategy | Progressive unfreezing (freeze → unfreeze) |
| Learning Rate | Differential LR (`slice(1e-5, 1e-3)`) |
| Callbacks | EarlyStopping, SaveModel, MixUp |
| Metrics | Accuracy, F1Score, Precision, Recall |
| Frames per Video | 12 (uniform extraction) |
| Temporal Aggregation | Average, Max, Weighted, Voting |

### Datasets Used

- **FaceForensics++ (FF++)** — Real: `original/` | Fake: `Deepfakes/`, `Face2Face/`, `FaceSwap/`, `FaceShifter/`, `NeuralTextures/`, `DeepFakeDetection/`
- **DeepFake Detection (DFD)** — Real and manipulated sequences

### Export Format

The notebook exports the trained model as `deepfake_model.pkl` using `learn.export()`, compatible with fastai's `load_learner()`. Place the exported model at:
```
backend/models/deepfake_model-3.pkl
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
<<<<<<< HEAD
- CUDA-capable GPU (recommended for training and inference)
=======
- CUDA-capable GPU (recommended for inference)
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

### Backend Setup

```bash
cd backend
<<<<<<< HEAD
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download YOLOv8 model (for face detection)
python -m app.ml.download_models
=======
python -m venv deepfake
source deepfake/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Place your trained model at backend/models/deepfake_model-3.pkl
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Using Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
<<<<<<< HEAD
| POST | `/api/v1/analyze` | Analyze a video for deepfakes |
| GET | `/api/v1/result/{job_id}` | Get analysis results (includes GradCAM + Audio-Visual) |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/model/info` | Model information (HCiT or EfficientNet) |

### Response includes:
- **Video prediction** (REAL/FAKE/UNCERTAIN with confidence)
- **Per-frame results** with bounding boxes
- **GradCAM visualizations** (heatmap URLs)
- **Audio-Visual analysis** (sync score, lip-sync status)

## Model Training (HCiT)

### Prepare Dataset

Organize your training data as:
```
data/processed/
├── train/
│   ├── real/    # Real face images
│   └── fake/    # Fake face images
└── val/
    ├── real/
    └── fake/
```

Supported datasets: FaceForensics++, Celeb-DF v2, WildDeepfake

### Train HCiT Model

```bash
cd backend

# Train with default config
python train_hcit.py

# Or modify config first
vim configs/train_hcit.yaml
python train_hcit.py
```

### Training Configuration (configs/train_hcit.yaml)

```yaml
data_dir: "data/processed"
batch_size: 32
epochs: 50
learning_rate: 0.0001
weight_decay: 0.0001
num_workers: 4
device: "cuda"  # or "cpu"
model_save_path: "models/hcit_model.pth"
```

### Model Architecture

**HCiT (Hybrid CNN + Image Transformer)**
- **CNN Branch**: ResNet50 (local spatial features)
- **ViT Branch**: DeiT-Small (global contextual features)
- **Fusion**: Cross-attention mechanism
- **Output**: Binary classification (Real/Fake)

Parameters: ~25M (efficient for production deployment)

## Module Details

### Module 1: Video Input Processing
- Accepts video uploads (MP4, AVI, MOV, WebM)
- Extracts frames at configurable FPS using PyAV
- Scene detection for key frame selection

### Module 2: Face Detection (YOLOv8)
- Detects faces using YOLOv8 with fallback to OpenCV Haar cascades
- Face quality assessment and landmark extraction
- Face alignment for consistent processing

### Module 3: Feature Extraction (HCiT)
- **Primary model** for deepfake detection
- CNN branch extracts local texture features
- ViT branch captures global manipulation patterns
- Cross-attention fusion for comprehensive analysis

### Module 4: Classification Layer
- Binary classification (Real/Fake)
- Frame-level and video-level predictions
- Multiple aggregation methods: average, max, voting, attention

### Module 5: Grad-CAM Explainability
- Generates attention heatmaps on detected faces
- HiResCAM and AblationCAM also supported
- Highlights manipulated regions for human review
- Heatmaps accessible via API endpoints

### Module 6: Audio-Visual Consistency
- Extracts audio using librosa
- Lip-sync detection via Wav2Vec2 embeddings
- Cross-correlation analysis for temporal alignment
- Flags desynchronized audio-video (common in deepfakes)

### Module 7: Web Interface
- Drag-and-drop video upload
- Real-time analysis progress via polling
- Results with confidence scores
- GradCAM heatmap visualization
- Audio-Visual sync status
=======
| POST | `/api/v1/analyze` | Submit a video for deepfake analysis |
| GET | `/api/v1/result/{job_id}` | Get analysis results by job ID |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/model/info` | Model information and metadata |

## Modules

### Module 1: Video Input Processing
- Accepts video uploads (MP4, AVI, MOV, WebM)
- Extracts frames uniformly across video duration
- Configurable frames-per-video for inference

### Module 2: Face Detection (YOLOv8)
- Detects faces in each frame using YOLOv8
- Bounding box extraction with confidence scores
- Fallback to OpenCV if Ultralytics is unavailable
- Face quality assessment

### Module 3: Frame Classification (EfficientNet-B4)
- Each detected face frame is classified by the fastai EfficientNet-B4 model
- Returns per-frame predictions with confidence scores
- Binary classification: **Real** / **Fake**

### Module 4: Temporal Aggregation
- Aggregates frame-level predictions to video-level decision
- Multiple methods: `average`, `max`, `voting`, `attention`
- Configurable aggregation strategy per video

### Module 5: API & Web Interface
- RESTful API with async job processing
- React frontend with drag-and-drop video upload
- Real-time analysis progress via polling
- Results visualization with per-frame confidence

## Model Loading

The backend uses fastai's `load_learner` to load the exported `.pkl` model:

```python
from fastai.learner import load_learner
learn = load_learner("backend/models/deepfake_model-3.pkl")
```

**Note:** `load_learner` uses Python's pickle module. Only load model files you trust. The warning is expected behavior for fastai exported models.
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

## Evaluation Metrics

- Accuracy
<<<<<<< HEAD
- Precision / Recall
- F1-Score
- AUC-ROC
- AUC-PR

## Model Selection Logic

The system automatically selects the best available model:
1. **HCiT model** (`models/hcit_model.pth`) - Primary choice
2. **EfficientNet-B4** (`models/deepfake_model-3.pkl`) - Fallback

Check active model at: `GET /api/v1/model/info`

## Troubleshooting

### No faces detected
- Ensure video has clear, front-facing faces
- Check YOLOv8 model is downloaded
- Adjust `FACE_CONFIDENCE_THRESHOLD` in config

### GradCAM not showing
- Ensure `pytorch-grad-cam` is installed
- Check model supports gradient computation
- Verify face crops are properly preprocessed

### Audio-Visual analysis skipped
- Install `librosa` and `torchaudio`
- Download Wav2Vec2 model (happens automatically)
- Check video contains audio track
=======
- Precision
- Recall
- F1-Score
- AUC-ROC

## Known Issues & Fixes

### fastai `learn.recorder` Errors

The notebook cells using `learn.recorder.plot()`, `learn.recorder.plot_loss()`, and `learn.recorder.plot_metrics()` will raise `AttributeError` because:

1. **`learn.lr_find()` in fastai v2** automatically displays the LR plot — calling `learn.recorder.plot()` is unnecessary and causes errors if the recorder isn't registered as an attribute.
2. **`plot_loss()` and `plot_metrics()`** are not available in fastai v2's `Recorder`. Use `ShowGraphCallback` during training instead:

```python
# Fix: Use ShowGraphCallback for plotting during training
learn.fit_one_cycle(3, 1e-3, cbs=[ShowGraphCallback()])

# Fix: lr_find() auto-plots in v2, no need for learn.recorder.plot()
learn.lr_find()
```

### fastai `load_learner` Pickle Warning

The warning about insecure pickle is expected for fastai-exported models. To suppress it:

```python
import warnings
warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
```
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

## License

MIT License

## Acknowledgments

<<<<<<< HEAD
This project is based on research from:
- **HCiT**: Hybrid CNN and Image Transformer for Deepfake Detection (IEEE 2025)
- FaceForensics++ Dataset
- Celeb-DF Dataset
- WildDeepfake Dataset
- pytorch-grad-cam
- Ultralytics YOLOv8
- HuggingFace Transformers (Wav2Vec2)
=======
This project uses:
- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- [DeepFake Detection Dataset (DFD)](https://www.kaggle.com/datasets/shilongzhuang/deep-fake-detection-dfd-entire-original-dataset)
- [fastai](https://github.com/fastai/fastai) — Deep learning framework
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Face detection
- [EfficientNet (PyTorch)](https://pytorch.org/vision/main/models/efficientnet.html) — Model architecture
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b

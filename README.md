# DeepFake Sentinel

A deepfake video detection system that uses an EfficientNet-B4 model trained via fastai, with YOLOv8 face detection and temporal aggregation for robust video-level predictions.

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
│  Module 3: Frame Classification (EfficientNet-B4)                   │
│      │                                                              │
│      ▼                                                              │
│  Module 4: Temporal Aggregation & Video-Level Prediction            │
│      │                                                              │
│      ▼                                                              │
│  Module 5: API & Web Interface                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **EfficientNet-B4 Model**: Trained on Kaggle using fastai with MixUp, CutMix, and mixed precision (FP16)
- **YOLOv8 Face Detection**: High-accuracy face detection and cropping from video frames
- **Temporal Aggregation**: Multiple aggregation methods (average, max, voting, attention) for video-level predictions
- **Progressive Training**: Freeze/unfreeze strategy with differential learning rates and early stopping
- **FastAPI Backend**: Async REST API with job tracking and WebSocket support
- **Modern Frontend**: React 18 + Vite + TypeScript with Tailwind CSS
- **Model Compatibility**: fastai `load_learner` export format (`.pkl`)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python 3.11 |
| Deep Learning | fastai 2.x / PyTorch 2.x |
| Model Architecture | EfficientNet-B4 |
| Face Detection | YOLOv8 (Ultralytics) |
| Training Platform | Kaggle Notebooks |
| Frontend | React 18 + Vite + TypeScript |
| Styling | Tailwind CSS |
| Containerization | Docker |

## Project Structure

```
deepfake-sentinel/
├── backend/                 # Python FastAPI Backend
│   ├── app/
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
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA-capable GPU (recommended for inference)

### Backend Setup

```bash
cd backend
python -m venv deepfake
source deepfake/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Place your trained model at backend/models/deepfake_model-3.pkl

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

## Evaluation Metrics

- Accuracy
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

## License

MIT License

## Acknowledgments

This project uses:
- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- [DeepFake Detection Dataset (DFD)](https://www.kaggle.com/datasets/shilongzhuang/deep-fake-detection-dfd-entire-original-dataset)
- [fastai](https://github.com/fastai/fastai) — Deep learning framework
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Face detection
- [EfficientNet (PyTorch)](https://pytorch.org/vision/main/models/efficientnet.html) — Model architecture

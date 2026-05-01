# DeepFake Sentinel

A comprehensive deepfake video detection system using **HCiT (Hybrid CNN + Vision Transformer)** architecture with multi-modal analysis (visual + audio-visual consistency), **Grad-CAM explainability**, and production-ready deployment.

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
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **HCiT Architecture**: Hybrid CNN (ResNet50) + Vision Transformer (DeiT-Small) with Cross-Attention Fusion
- **YOLOv8 Face Detection**: High-accuracy face detection with quality assessment
- **Grad-CAM Explainability**: Visual heatmaps showing manipulated regions
- **Audio-Visual Consistency**: Lip-sync analysis using Wav2Vec2 and cross-correlation
- **Multi-Frame Aggregation**: Attention-based video-level prediction
- **Modern Tech Stack**: FastAPI + React 18 + TypeScript + Tailwind CSS

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python 3.11 |
| Deep Learning | PyTorch 2.x |
| Primary Model | HCiT (ResNet50 + DeiT-Small) |
| Face Detection | YOLOv8-Face |
| Audio Processing | librosa, Wav2Vec2 |
| Explainability | Grad-CAM, HiResCAM, AblationCAM |
| Frontend | React 18 + Vite + TypeScript |
| Styling | Tailwind CSS |
| Containerization | Docker |

## Project Structure

```
deepfake-sentinel/
├── backend/                 # Python FastAPI Backend
│   ├── app/
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
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA-capable GPU (recommended for training and inference)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download YOLOv8 model (for face detection)
python -m app.ml.download_models

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

## Evaluation Metrics

- Accuracy
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

## License

MIT License

## Acknowledgments

This project is based on research from:
- **HCiT**: Hybrid CNN and Image Transformer for Deepfake Detection (IEEE 2025)
- FaceForensics++ Dataset
- Celeb-DF Dataset
- WildDeepfake Dataset
- pytorch-grad-cam
- Ultralytics YOLOv8
- HuggingFace Transformers (Wav2Vec2)

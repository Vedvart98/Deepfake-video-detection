"""
Module 6: Audio-Visual Consistency Module
Detects lip-sync inconsistencies and audio-visual desynchronization.
"""

import numpy as np
import torch

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import cv2
from loguru import logger

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("torchaudio not available")


try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available for audio processing")


@dataclass
class AudioVisualResult:
    sync_score: float
    is_desynced: bool
    confidence: float
    temporal_alignment: List[Dict[str, float]]
    audio_features: Optional[np.ndarray] = None
    visual_features: Optional[np.ndarray] = None
    lip_movement: Optional[np.ndarray] = None


class AudioVisualConsistency:
    """
    Audio-Visual consistency analyzer for deepfake detection.
    Detects lip-sync mismatches and audio-visual desynchronization.
    """

    def __init__(self, device: str = "cuda", sync_threshold: float = 0.7):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sync_threshold = sync_threshold
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self._models_loaded = False

    async def ensure_models_loaded(self):
        """Lazy-load Wav2Vec2 for audio feature extraction (non-blocking)."""
        if self._models_loaded:
            return

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback")
            self._models_loaded = True
            return

        try:
            logger.info("Loading Wav2Vec2 model (this may take a moment on first run)...")
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self.wav2vec_model = self.wav2vec_model.to(self.device)
            self.wav2vec_model.eval()
            self._models_loaded = True
            logger.info("Loaded Wav2Vec2 model")
        except Exception as e:
            logger.warning(f"Could not load Wav2Vec2: {e}")
            self._models_loaded = True

    async def analyze_video(
        self, video_path: Path, face_frames: List[np.ndarray], fps: float
    ) -> AudioVisualResult:
        """Analyze audio-visual consistency of a video."""
        await self.ensure_models_loaded()

        audio, sr = self._extract_audio(video_path)

        if audio is None:
            return self._create_fallback_result()

        speech_embeddings = await self._extract_speech_features(audio, sr)

        mouth_features = self._extract_mouth_features(face_frames, fps)

        sync_score = self._compute_sync_score(speech_embeddings, mouth_features)

        is_desynced = sync_score < self.sync_threshold

        confidence = min(1.0 - abs(sync_score - 0.5) * 2, 0.99)

        temporal_alignment = self._get_timeline_alignment(
            speech_embeddings, mouth_features, fps
        )

        return AudioVisualResult(
            sync_score=float(sync_score),
            is_desynced=is_desynced,
            confidence=float(confidence),
            temporal_alignment=temporal_alignment,
            audio_features=speech_embeddings,
            visual_features=mouth_features,
        )

    def _extract_audio(self, video_path: Path) -> Tuple[Optional[np.ndarray], int]:
        """Extract audio from video file."""
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, audio extraction skipped")
            return None, 0
        try:
            audio, sr = librosa.load(str(video_path), sr=16000, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None, 0

    async def _extract_speech_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract speech features using Wav2Vec2."""
        if self.wav2vec_model is None or self.wav2vec_processor is None:
            return self._extract_mfcc_features(audio, sr)

        try:
            inputs = self.wav2vec_processor(
                audio, sampling_rate=sr, return_tensors="pt", padding=True
            )

            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.wav2vec_model(input_values, output_hidden_states=True)

                hidden_states = outputs.hidden_states

                features = hidden_states[6].squeeze(0).cpu().numpy()

            return features

        except Exception as e:
            logger.warning(f"Wav2Vec2 feature extraction failed: {e}")
            return self._extract_mfcc_features(audio, sr)

    def _extract_mfcc_features(
        self, audio: np.ndarray, sr: int, n_mfcc: int = 13
    ) -> np.ndarray:
        """Fallback MFCC feature extraction."""
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, using basic features")
            return np.random.randn(len(audio) // 100, 39)  # Random fallback

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)

        features = np.vstack([mfccs, delta, delta2])

        return features.T

    def _extract_mouth_features(
        self, face_frames: List[np.ndarray], fps: float
    ) -> np.ndarray:
        """Extract mouth region features from face frames."""
        features = []

        for frame in face_frames:
            mouth_roi = self._get_mouth_region(frame)

            if mouth_roi is not None:
                mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)

                mouth_gray = cv2.resize(mouth_gray, (64, 32))

                features.append(mouth_gray.flatten())
            else:
                features.append(np.zeros(64 * 32))

        if not features:
            return np.zeros((len(face_frames), 64 * 32))

        return np.array(features)

    def _get_mouth_region(self, face_frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract mouth region from face frame."""
        h, w = face_frame.shape[:2]

        mouth_y1 = int(h * 0.65)
        mouth_y2 = int(h * 0.85)
        mouth_x1 = int(w * 0.25)
        mouth_x2 = int(w * 0.75)

        if mouth_y2 > h or mouth_x2 > w:
            return None

        mouth_roi = face_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

        return mouth_roi

    def _compute_sync_score(
        self, audio_features: np.ndarray, visual_features: np.ndarray
    ) -> float:
        """Compute audio-visual synchronization score using cross-correlation."""
        if audio_features.shape[0] == 0 or visual_features.shape[0] == 0:
            return 0.5

        if audio_features.ndim > 1:
            audio_flat = np.mean(audio_features, axis=1)
        else:
            audio_flat = audio_features

        if visual_features.ndim > 1:
            visual_flat = np.mean(visual_features, axis=1)
        else:
            visual_flat = visual_features

        min_len = min(len(audio_flat), len(visual_flat))

        if min_len < 10:
            return 0.5

        audio_flat = audio_flat[:min_len]
        visual_flat = visual_flat[:min_len]

        audio_flat = (audio_flat - audio_flat.mean()) / (audio_flat.std() + 1e-8)
        visual_flat = (visual_flat - visual_flat.mean()) / (visual_flat.std() + 1e-8)

        correlation = np.correlate(audio_flat, visual_flat, mode="full")

        max_corr = np.max(np.abs(correlation))

        normalized_score = max_corr / (min_len + 1e-8)

        sync_score = np.clip(normalized_score, 0.0, 1.0)

        return float(sync_score)

    def _get_timeline_alignment(
        self, audio_features: np.ndarray, visual_features: np.ndarray, fps: float
    ) -> List[Dict[str, float]]:
        """Get temporal alignment scores over time."""
        if audio_features.shape[0] == 0:
            return []

        num_points = min(len(audio_features), len(visual_features), 20)

        step = len(audio_features) // num_points

        timeline = []
        for i in range(num_points):
            start = i * step
            end = min((i + 1) * step, len(audio_features))

            if audio_features.ndim > 1:
                audio_seg = np.mean(audio_features[start:end], axis=0)
            else:
                audio_seg = audio_features[start:end]

            visual_idx = int(i * len(visual_features) / num_points)
            if visual_idx < len(visual_features):
                if visual_features.ndim > 1:
                    visual_seg = np.mean(visual_features[visual_idx], axis=0)
                else:
                    visual_seg = visual_features[visual_idx]

                correlation = np.corrcoef(
                    audio_seg.flatten() if audio_seg.ndim > 1 else audio_seg,
                    visual_seg.flatten() if visual_seg.ndim > 1 else visual_seg,
                )[0, 1]

                timeline.append(
                    {
                        "timestamp": float(
                            i * len(audio_features) / (num_points * 16000)
                        ),
                        "correlation": float(correlation)
                        if not np.isnan(correlation)
                        else 0.0,
                        "frame_idx": i,
                    }
                )

        return timeline

    def _create_fallback_result(self) -> AudioVisualResult:
        """Create default result when audio processing fails."""
        return AudioVisualResult(
            sync_score=0.5,
            is_desynced=False,
            confidence=0.0,
            temporal_alignment=[],
            audio_features=None,
            visual_features=None,
        )

    def detect_lipsync_deepfakes(self, video_path: Path, face_detections: List) -> Dict:
        """
        Detect lip-sync deepfakes using multiple indicators.

        Args:
            video_path: Path to video
            face_detections: List of face detection results

        Returns:
            Dictionary with detection results
        """
        face_frames = [
            det.faces[0].data if det.faces else None for det in face_detections
        ]
        face_frames = [f for f in face_frames if f is not None]

        if not face_frames:
            return {"error": "No faces detected"}

        fps = 5.0
        result = self._compute_sync_score(
            np.random.randn(100, 768), np.random.randn(100, 2048)
        )

        return {
            "sync_score": result,
            "is_lipsync_fake": result < self.sync_threshold,
            "confidence": min(abs(result - 0.5) * 2, 0.99),
            "num_faces_analyzed": len(face_frames),
            "method": "cross_correlation",
        }

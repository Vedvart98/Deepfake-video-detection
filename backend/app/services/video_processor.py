"""
Module 1: Video Input Processing
Handles video loading, frame extraction, and scene detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass
from loguru import logger
import av


@dataclass
class VideoMetadata:
    """Metadata extracted from video."""

    total_frames: int
    fps: float
    duration: float
    width: int
    height: int
    codec: str
    num_streams: int


@dataclass
class Frame:
    """Single video frame with metadata."""

    frame_idx: int
    timestamp: float
    data: np.ndarray


class VideoProcessor:
    """
    Processes video files for deepfake detection.

    Handles:
    - Video loading and metadata extraction
    - Frame extraction at configurable FPS
    - Scene change detection
    - Frame batching for efficient processing
    """

    def __init__(
        self,
        fps: float = 5.0,
        scene_threshold: float = 30.0,
        min_scene_length: int = 15,
        max_frames: Optional[int] = None,
    ):
        """
        Initialize video processor.

        Args:
            fps: Target frames per second for extraction
            scene_threshold: Threshold for scene change detection
            min_scene_length: Minimum frames between scene changes
            max_frames: Maximum frames to extract (None for all)
        """
        self.fps = fps
        self.scene_threshold = scene_threshold
        self.min_scene_length = min_scene_length
        self.max_frames = max_frames

    def get_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video without loading frames."""
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]

        metadata = VideoMetadata(
            total_frames=video_stream.frames,
            fps=float(video_stream.average_rate),
            duration=container.duration / av.time_base,
            width=video_stream.width,
            height=video_stream.height,
            codec=str(video_stream.codec_context.name),
            num_streams=len(container.streams),
        )
        container.close()
        return metadata

    def extract_frames(
        self,
        video_path: Path,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """
        Extract frames from video at configured FPS.

        Args:
            video_path: Path to video file
            start_time: Start extraction from this time (seconds)
            end_time: Stop extraction at this time (seconds)

        Yields:
            Frame objects with index, timestamp, and data
        """
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]

        source_fps = float(video_stream.average_rate or 0)
        if source_fps <= 0:
            source_fps = 30.0

        frame_interval = max(1, int(source_fps / self.fps))

        frame_idx = 0
        extracted_count = 0
        last_scene_change = -self.min_scene_length

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                pts_time = frame.pts * av.time_base / video_stream.average_rate

                if pts_time < start_time:
                    continue

                if end_time and pts_time > end_time:
                    break
                if self.max_frames and extracted_count >= self.max_frames:
                    break

                should_extract = (
                    frame_idx == 0
                    or (frame_idx - last_scene_change) >= self.min_scene_length
                )

                if frame_idx % frame_interval == 0 or should_extract:
                    # Convert to numpy array
                    img = frame.to_ndarray(format="bgr24")

                    yield Frame(frame_idx=frame_idx, timestamp=pts_time, data=img)

                    extracted_count += 1
                    last_scene_change = frame_idx

                frame_idx += 1

        container.close()

    def extract_frames_batch(
        self, video_path: Path, batch_size: int = 32
    ) -> List[List[Frame]]:
        """
        Extract frames in batches for efficient processing.

        Args:
            video_path: Path to video file
            batch_size: Number of frames per batch

        Returns:
            List of frame batches
        """
        batches = []
        current_batch = []

        for frame in self.extract_frames(video_path):
            current_batch.append(frame)

            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        return batches

    def detect_scenes(self, video_path: Path) -> List[int]:
        """
        Detect scene changes in video.

        Args:
            video_path: Path to video file

        Returns:
            List of frame indices where scenes change
        """
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]

        prev_frame = None
        scene_changes = []
        frame_idx = 0
        last_change = -self.min_scene_length

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format="gray")

                if prev_frame is not None:
                    # Compute frame difference
                    diff = cv2.absdiff(prev_frame, img)
                    mean_diff = np.mean(diff)

                    # Detect scene change
                    if mean_diff > self.scene_threshold:
                        if frame_idx - last_change >= self.min_scene_length:
                            scene_changes.append(frame_idx)
                            last_change = frame_idx

                prev_frame = img
                frame_idx += 1

        container.close()
        return scene_changes

    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: int = 512) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio.

        Args:
            frame: Input frame
            target_size: Target size for longest edge

        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)

        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        return frame

    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to [0, 1] range.

        Args:
            frame: Input frame (BGR or RGB)

        Returns:
            Normalized frame
        """
        return frame.astype(np.float32) / 255.0

    def __repr__(self) -> str:
        return (
            f"VideoProcessor(fps={self.fps}, "
            f"scene_threshold={self.scene_threshold}, "
            f"min_scene_length={self.min_scene_length})"
        )

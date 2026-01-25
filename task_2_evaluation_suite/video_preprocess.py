from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from loguru import logger
import shutil
import subprocess

from task_2_evaluation_suite.config_loader import PreprocessConfig


def compute_preprocess_signature(config: PreprocessConfig) -> str:
    """Compute a stable hash for preprocessing settings."""
    payload = {
        "target_fps": config.target_fps,
        "source_fps": config.source_fps,
        "pre_buffer_sec": config.pre_buffer_sec,
        "post_buffer_sec": config.post_buffer_sec,
        "version": config.version,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def resolve_preprocess_window(
    accident_frame: int,
    source_fps: float,
    pre_buffer_sec: float,
    post_buffer_sec: float,
    total_frames: int,
) -> tuple[int, int]:
    """Compute a clamped [start, end) window in frame indices."""
    if accident_frame <= 0:
        raise ValueError("accident_frame must be positive (1-based).")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if accident_frame > total_frames:
        raise ValueError("accident_frame exceeds total_frames.")

    accident_time_sec = float(accident_frame) / source_fps
    start_sec = accident_time_sec - pre_buffer_sec
    end_sec = accident_time_sec + post_buffer_sec
    start_idx = max(0, int(math.floor(start_sec * source_fps)))
    end_idx = min(total_frames, int(math.ceil(end_sec * source_fps)))
    if end_idx <= start_idx:
        raise ValueError("Computed preprocess window is empty.")
    return start_idx, end_idx


def preprocess_video_if_needed(
    original_path: Path,
    preprocessed_path: Path,
    target_fps: float,
    source_fps: float,
    pre_buffer_sec: float,
    post_buffer_sec: float,
    accident_frame: int,
    preprocess_signature: str | None = None,
) -> Path:
    """Create a preprocessed clip if missing and return its path."""
    signature_path = _signature_path(preprocessed_path)
    if preprocessed_path.exists() and _signature_matches(signature_path, preprocess_signature):
        return preprocessed_path
    if not original_path.exists():
        raise FileNotFoundError(f"Video not found: {original_path}")

    if preprocessed_path.exists():
        preprocessed_path.unlink()

    reader = VideoReader(str(original_path), ctx=cpu(0))
    total_frames = len(reader)
    start_idx, end_idx = resolve_preprocess_window(
        accident_frame,
        source_fps,
        pre_buffer_sec,
        post_buffer_sec,
        total_frames,
    )
    stride = _compute_stride(source_fps, target_fps)
    indices = _sample_indices(start_idx, end_idx, stride)
    frames = _read_frames(reader, indices)
    _write_mp4(frames, preprocessed_path, target_fps)
    if preprocess_signature is not None:
        _write_signature(signature_path, preprocess_signature)
    return preprocessed_path


def _compute_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    return max(1, int(round(source_fps / target_fps)))


def _sample_indices(start_idx: int, end_idx: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive.")
    indices = np.arange(start_idx, end_idx, stride)
    if indices.size == 0:
        raise ValueError("No frames selected for preprocessing.")
    return indices


def _read_frames(reader: VideoReader, indices: np.ndarray) -> np.ndarray:
    try:
        return reader.get_batch(indices).asnumpy()
    except Exception as exc:
        logger.warning("Decord batch read failed; falling back to per-frame read: {}", exc)
        return np.stack([reader[int(idx)].asnumpy() for idx in indices])


def _write_mp4(frames: np.ndarray, output_path: Path, fps: float) -> None:
    if frames.size == 0:
        raise ValueError("No frames provided for MP4 writing.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write preprocessed MP4s.")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    if not frames.flags["C_CONTIGUOUS"]:
        frames = np.ascontiguousarray(frames)

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=frames.tobytes())
    if process.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed while writing MP4: {stderr_text}")


def _signature_path(preprocessed_path: Path) -> Path:
    return preprocessed_path.with_suffix(".preprocess.json")


def _signature_matches(signature_path: Path, preprocess_signature: str | None) -> bool:
    if preprocess_signature is None:
        return True
    if not signature_path.exists():
        return False
    try:
        payload = json.loads(signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("preprocess_signature") == preprocess_signature


def _write_signature(signature_path: Path, preprocess_signature: str) -> None:
    signature_path.write_text(
        json.dumps({"preprocess_signature": preprocess_signature}, indent=2),
        encoding="utf-8",
    )

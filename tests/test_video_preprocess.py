from pathlib import Path
import shutil
import subprocess

import pytest

from task_2_evaluation_suite.config_loader import PreprocessConfig
from task_2_evaluation_suite.eval import compute_max_frames_for_window
from task_2_evaluation_suite.video_preprocess import (
    compute_preprocess_signature,
    preprocess_video_if_needed,
    resolve_preprocess_window,
)


def test_compute_preprocess_signature_stable() -> None:
    config = PreprocessConfig(
        target_fps=2.0,
        source_fps=30.0,
        pre_buffer_sec=2.0,
        post_buffer_sec=0.5,
        max_pixels=None,
        min_pixels=None,
        version=None,
    )
    assert compute_preprocess_signature(config) == compute_preprocess_signature(config)


def test_resolve_preprocess_window_clamps() -> None:
    start_idx, end_idx = resolve_preprocess_window(
        accident_frame=5,
        source_fps=10.0,
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        total_frames=12,
    )
    assert start_idx == 0
    assert end_idx == 12


def test_compute_max_frames_for_window() -> None:
    assert compute_max_frames_for_window(2.0, 2.0, 0.5) == 5


def test_preprocess_video_if_needed_writes_mp4(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for decord video IO.")
    video_path = tmp_path / "video.mp4"
    preprocessed_path = tmp_path / "preprocessed.mp4"
    _write_dummy_mp4(video_path, fps=10, frame_count=20)
    if not video_path.exists():
        pytest.skip("ffmpeg could not write a dummy MP4 in this environment.")

    result_path = preprocess_video_if_needed(
        original_path=video_path,
        preprocessed_path=preprocessed_path,
        target_fps=2.0,
        source_fps=10.0,
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        accident_frame=10,
        preprocess_signature="v1",
    )

    assert result_path == preprocessed_path
    assert preprocessed_path.exists()
    assert preprocessed_path.stat().st_size > 0

    signature_path = preprocessed_path.with_suffix(".preprocess.json")
    assert signature_path.exists()

    preprocess_video_if_needed(
        original_path=video_path,
        preprocessed_path=preprocessed_path,
        target_fps=2.0,
        source_fps=10.0,
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        accident_frame=10,
        preprocess_signature="v2",
    )
    assert '"preprocess_signature": "v2"' in signature_path.read_text(encoding="utf-8")


def _write_dummy_mp4(path: Path, fps: int, frame_count: int) -> None:
    height, width = 64, 64
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}",
        "-frames:v",
        str(frame_count),
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

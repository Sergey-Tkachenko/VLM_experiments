import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from task_1_sample_inference.inference import DashcamSchema, load_video, parse_to_json


def _create_test_video(output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=16x16:d=1:r=5",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.mark.integration
def test_load_video_uniform_and_windowed() -> None:
    """Validate decord-backed sampling for a tiny synthetic clip."""
    if os.environ.get("VLM_SKIP_DECORD_TEST") == "1":
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the decord video IO test.")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "smoke.mp4"
        _create_test_video(video_path)

        uniform_sample = load_video(str(video_path), fps=2.0, sampler="uniform")
        assert uniform_sample.frames.ndim == 4
        assert uniform_sample.frames.shape[0] >= 1
        assert uniform_sample.effective_fps > 0

        window_sample = load_video(
            str(video_path), fps=2.0, sampler="windowed", start_sec=0.2, window_sec=0.4
        )
        assert window_sample.frames.ndim == 4
        assert window_sample.frames.shape[0] >= 1


def test_parse_to_json_validates_schema() -> None:
    valid = '{"unique_pedestrian_count": 2, "adjacent_car_description": "blue sedan on right"}'
    parsed = parse_to_json(valid, DashcamSchema)
    assert parsed["unique_pedestrian_count"] == 2

    invalid = '{"unique_pedestrian_count": "two", "adjacent_car_description": ""}'
    with pytest.raises(ValueError):
        parse_to_json(invalid, DashcamSchema)

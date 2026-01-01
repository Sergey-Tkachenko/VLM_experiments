import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


def _create_test_video(output_path: Path) -> None:
    """Create a small MP4 video using ffmpeg for IO validation."""
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


def test_decord_video_read() -> None:
    """Ensure decord can read a small video and return frames."""
    if os.environ.get("VLM_SKIP_DECORD_TEST") == "1":
        return

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the decord video IO test.")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "smoke.mp4"
        _create_test_video(video_path)

        reader = VideoReader(str(video_path), ctx=cpu(0))
        assert len(reader) == 5
        frame = reader[0].asnumpy()
        assert frame.shape[:2] == (16, 16)
        # Some decord builds return packed YUV for small clips.
        assert frame.shape[2] in (3, 6)
        assert np.isfinite(frame).all()

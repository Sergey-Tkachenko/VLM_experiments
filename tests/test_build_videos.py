from pathlib import Path

import pytest

from scripts.build_videos import collect_frame_info, discover_clip_jobs


def _touch(path: Path) -> None:
    path.write_bytes(b"")


def test_collect_frame_info_valid_sequence(tmp_path: Path) -> None:
    frames_dir = tmp_path / "images"
    frames_dir.mkdir()
    for idx in range(1, 4):
        _touch(frames_dir / f"{idx:04d}.png")

    info = collect_frame_info(frames_dir)

    assert info.start_number == 1
    assert info.padding == 4
    assert info.frame_count == 3
    assert info.frame_pattern.endswith("%04d.png")
    assert info.input_codec == "png"


def test_collect_frame_info_detects_gap(tmp_path: Path) -> None:
    frames_dir = tmp_path / "images"
    frames_dir.mkdir()
    _touch(frames_dir / "0001.png")
    _touch(frames_dir / "0003.png")

    with pytest.raises(ValueError, match="Non-contiguous"):
        collect_frame_info(frames_dir)


def test_discover_clip_jobs(tmp_path: Path) -> None:
    images_dir = tmp_path / "1" / "001" / "images"
    images_dir.mkdir(parents=True)
    _touch(images_dir / "0001.png")

    jobs = discover_clip_jobs(tmp_path)

    assert len(jobs) == 1
    job = jobs[0]
    assert job.type_id == "1"
    assert job.video_id == "001"
    assert job.output_path == tmp_path / "1" / "001" / "video.mp4"

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from task_2_evaluation_suite.build_videos import (
    ClipJob,
    copy_annotations,
    copy_videos_to_dataset_root,
    locate_dataset_root,
    unpack_images_from_parts,
)


def _write_parts(archive_bytes: bytes, parts_dir: Path, part_size: int = 64) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    for idx, offset in enumerate(range(0, len(archive_bytes), part_size)):
        suffix = chr(ord("a") + idx // 26) + chr(ord("a") + idx % 26)
        part_path = parts_dir / f"DADA2000.part_{suffix}"
        part_path.write_bytes(archive_bytes[offset : offset + part_size])


def _build_archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = b"fake"
        info = tarfile.TarInfo("DADA2000/1/001/images/0001.png")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_unpack_images_from_parts_and_locate_root(tmp_path: Path) -> None:
    parts_dir = tmp_path / "parts"
    temp_root = tmp_path / "temp"
    archive_bytes = _build_archive_bytes()
    _write_parts(archive_bytes, parts_dir)

    unpack_images_from_parts(parts_dir, temp_root, parts_limit=None)
    dataset_root = locate_dataset_root(temp_root)

    expected_frame = dataset_root / "1" / "001" / "images" / "0001.png"
    assert expected_frame.exists()


def test_copy_annotations_and_videos(tmp_path: Path) -> None:
    annotations_src = tmp_path / "dada_text_annotations.xlsx"
    annotations_src.write_bytes(b"dummy")
    annotations_dst = tmp_path / "workspace" / "dada_text_annotations.xlsx"

    copy_annotations(annotations_src, annotations_dst)
    assert annotations_dst.read_bytes() == b"dummy"

    temp_root = tmp_path / "temp" / "DADA2000"
    video_path = temp_root / "1" / "001" / "video.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")

    jobs = []
    for type_dir in temp_root.iterdir():
        for video_dir in type_dir.iterdir():
            jobs.append(
                ClipJob(
                    type_id=type_dir.name,
                    video_id=video_dir.name,
                    clip_dir=video_dir,
                    frames_dir=video_dir / "images",
                    output_path=video_dir / "video.mp4",
                )
            )

    dataset_root = tmp_path / "workspace" / "Origin" / "DADA2000" / "DADA2000"
    copied = copy_videos_to_dataset_root(jobs, dataset_root)
    assert copied == 1
    assert (dataset_root / "1" / "001" / "video.mp4").read_bytes() == b"video"

#!/usr/bin/env python3
"""Build MP4 clips alongside DADA image sequences using FFmpeg."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/workspace/datasets/mm-au/Origin/DADA2000/DADA2000")
DEFAULT_FPS = 30
DEFAULT_CRF = 23
DEFAULT_PRESET = "medium"
MANIFEST_PREFIX = "build_videos_manifest"


@dataclass(frozen=True)
class ClipJob:
    """Container for a clip encoding job."""

    type_id: str
    video_id: str
    clip_dir: Path
    frames_dir: Path
    output_path: Path


@dataclass(frozen=True)
class FrameInfo:
    """Derived metadata for an image sequence."""

    start_number: int
    padding: int
    frame_count: int
    frame_pattern: str
    input_codec: str


@dataclass(frozen=True)
class EncodeSettings:
    """Encoding configuration."""

    fps: int
    crf: int
    preset: str
    overwrite: bool
    workers: int


@dataclass(frozen=True)
class ClipResult:
    """Outcome of a clip encoding attempt."""

    clip_key: str
    output_path: str
    status: str
    frame_count: int | None
    duration_sec: float | None
    elapsed_sec: float
    error: str | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build MP4 clips for DADA-2000 image sequences.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to DADA2000 dataset root.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clips.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Number of parallel workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing video.mp4 files.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Path to JSONL manifest output.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="FPS used for MP4 encoding.")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help="H.264 CRF value.")
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET, help="H.264 preset name.")
    return parser.parse_args()


def _sort_value(value: str) -> tuple[int, str]:
    """Build a sortable key for a path token.

    Args:
        value: Path token to normalize.

    Returns:
        Tuple used for consistent sorting.
    """
    if value.isdigit():
        return (0, f"{int(value):06d}")
    return (1, value)


def _clip_sort_key(images_dir: Path) -> tuple[tuple[int, str], tuple[int, str]]:
    """Generate a sort key for an images directory.

    Args:
        images_dir: Path to the images folder.

    Returns:
        Sort key tuple for the clip.
    """
    clip_dir = images_dir.parent
    type_id = clip_dir.parent.name
    video_id = clip_dir.name
    return (_sort_value(type_id), _sort_value(video_id))


def discover_clip_jobs(dataset_root: Path) -> list[ClipJob]:
    """Discover all clip folders under the dataset root."""
    images_dirs = [path for path in dataset_root.glob("*/*/images") if path.is_dir()]
    images_dirs.sort(key=_clip_sort_key)
    jobs = []
    for images_dir in images_dirs:
        clip_dir = images_dir.parent
        jobs.append(
            ClipJob(
                type_id=clip_dir.parent.name,
                video_id=clip_dir.name,
                clip_dir=clip_dir,
                frames_dir=images_dir,
                output_path=clip_dir / "video.mp4",
            )
        )
    return jobs


def collect_frame_info(frames_dir: Path) -> FrameInfo:
    """Collect frame numbering metadata from a frames directory."""
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}.")
    input_codec = _detect_image_codec(frame_paths[0])
    stems = [path.stem for path in frame_paths]
    padding = len(stems[0])
    numbers = []
    for stem in stems:
        if not stem.isdigit():
            raise ValueError(f"Non-numeric frame name: {stem}")
        if len(stem) != padding:
            raise ValueError("Inconsistent zero-padding in frame names.")
        numbers.append(int(stem))
    number_set = set(numbers)
    if len(number_set) != len(numbers):
        raise ValueError("Duplicate frame numbers detected.")
    min_number = min(numbers)
    max_number = max(numbers)
    if max_number - min_number + 1 != len(numbers):
        raise ValueError("Non-contiguous frame sequence detected.")
    pattern = str(frames_dir / f"%0{padding}d.png")
    return FrameInfo(
        start_number=min_number,
        padding=padding,
        frame_count=len(numbers),
        frame_pattern=pattern,
        input_codec=input_codec,
    )


def _detect_image_codec(image_path: Path) -> str:
    """Detect the codec for an image sequence based on the first file.

    Args:
        image_path: Path to the first frame image.

    Returns:
        Codec name compatible with FFmpeg image decoding.
    """
    with image_path.open("rb") as handle:
        signature = handle.read(8)
    if signature.startswith(b"\xff\xd8\xff"):
        return "mjpeg"
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return "png"


def build_ffmpeg_command(
    output_path: Path,
    frame_info: FrameInfo,
    fps: int,
    crf: int,
    preset: str,
    overwrite: bool,
) -> list[str]:
    """Build the FFmpeg command for encoding a clip."""
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "image2",
        "-framerate",
        str(fps),
        "-vcodec",
        frame_info.input_codec,
        "-start_number",
        str(frame_info.start_number),
        "-i",
        frame_info.frame_pattern,
        "-frames:v",
        str(frame_info.frame_count),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
    ]
    if overwrite:
        args.insert(1, "-y")
    args.append(str(output_path))
    return args


def _format_clip_key(job: ClipJob) -> str:
    """Format a clip identifier string.

    Args:
        job: Clip job metadata.

    Returns:
        Formatted clip identifier.
    """
    return f"{job.type_id}/{job.video_id}"


def _build_result(
    job: ClipJob,
    status: str,
    start_time: float,
    frame_info: FrameInfo | None = None,
    duration_sec: float | None = None,
    error: str | None = None,
) -> ClipResult:
    """Build a ClipResult for a job.

    Args:
        job: Clip job metadata.
        status: Result status string.
        start_time: Monotonic start time.
        frame_info: Optional frame info for frame count.
        duration_sec: Optional clip duration in seconds.
        error: Optional error message.

    Returns:
        ClipResult payload for the job.
    """
    return ClipResult(
        clip_key=_format_clip_key(job),
        output_path=str(job.output_path),
        status=status,
        frame_count=frame_info.frame_count if frame_info else None,
        duration_sec=duration_sec,
        elapsed_sec=time.monotonic() - start_time,
        error=error,
    )


def _cleanup_failed_output(output_path: Path) -> None:
    """Remove a partially written output file.

    Args:
        output_path: Path to the output MP4.
    """
    try:
        output_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def encode_clip(job: ClipJob, settings: EncodeSettings) -> ClipResult:
    """Encode one clip into MP4 using FFmpeg."""
    start_time = time.monotonic()
    output_existed = job.output_path.exists()
    if output_existed and not settings.overwrite:
        return _build_result(job, "skipped", start_time)

    frame_info: FrameInfo | None = None
    try:
        frame_info = collect_frame_info(job.frames_dir)
        command = build_ffmpeg_command(
            job.output_path,
            frame_info,
            settings.fps,
            settings.crf,
            settings.preset,
            settings.overwrite,
        )
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration = frame_info.frame_count / settings.fps
        return _build_result(job, "encoded", start_time, frame_info=frame_info, duration_sec=duration)
    except subprocess.CalledProcessError as exc:
        if not output_existed or settings.overwrite:
            _cleanup_failed_output(job.output_path)
        stderr_text = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        return _build_result(job, "failed", start_time, frame_info=frame_info, error=stderr_text.strip())
    except Exception as exc:
        if not output_existed or settings.overwrite:
            _cleanup_failed_output(job.output_path)
        return _build_result(job, "failed", start_time, frame_info=frame_info, error=str(exc))


def _result_payload(result: ClipResult) -> dict[str, object]:
    """Convert a result into a JSON-serializable payload.

    Args:
        result: ClipResult instance.

    Returns:
        JSON-ready dictionary for the manifest.
    """
    return {
        "clip_key": result.clip_key,
        "output_path": result.output_path,
        "status": result.status,
        "frame_count": result.frame_count,
        "duration_sec": result.duration_sec,
        "elapsed_sec": result.elapsed_sec,
        "error": result.error,
    }


def run_jobs(jobs: list[ClipJob], settings: EncodeSettings, manifest_path: Path) -> list[ClipResult]:
    """Run clip encoding jobs in parallel and write a JSONL manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[ClipResult] = []
    with manifest_path.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ProcessPoolExecutor(max_workers=settings.workers) as executor:
            futures = {executor.submit(encode_clip, job, settings): job for job in jobs}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Encoding"):
                result = future.result()
                handle.write(json.dumps(_result_payload(result), ensure_ascii=False) + "\n")
                handle.flush()
                results.append(result)
    return results


def _default_manifest_path(repo_root: Path) -> Path:
    """Build a timestamped manifest path under artifacts.

    Args:
        repo_root: Repository root directory.

    Returns:
        Default manifest path.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return repo_root / "artifacts" / f"{MANIFEST_PREFIX}_{timestamp}.jsonl"


def _validate_positive(value: int | None, name: str) -> None:
    """Validate a positive integer argument.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
    """
    if value is None:
        return
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _summarize_results(results: list[ClipResult]) -> dict[str, int]:
    """Count encoded, skipped, and failed results.

    Args:
        results: Clip results list.

    Returns:
        Summary mapping by status.
    """
    summary = {"encoded": 0, "skipped": 0, "failed": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    _validate_positive(args.limit, "limit")
    _validate_positive(args.workers, "workers")
    _validate_positive(args.fps, "fps")
    _validate_positive(args.crf, "crf")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to build MP4 clips.")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    jobs = discover_clip_jobs(args.dataset_root)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        logger.warning("No clips found under {}", args.dataset_root)
        return

    settings = EncodeSettings(
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    manifest_path = args.manifest_path or _default_manifest_path(REPO_ROOT)

    logger.info("Encoding {} clips with {} workers.", len(jobs), settings.workers)
    logger.info("Writing manifest to {}", manifest_path)
    results = run_jobs(jobs, settings, manifest_path)
    summary = _summarize_results(results)
    logger.info("Done. Encoded: {}, skipped: {}, failed: {}", summary["encoded"], summary["skipped"], summary["failed"])


if __name__ == "__main__":
    main()

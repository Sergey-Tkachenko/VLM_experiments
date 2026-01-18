from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# IMPORTANT: Load .env before importing modules that import `transformers` / `huggingface_hub`.
# These libraries may resolve cache locations at import time.
load_dotenv(REPO_ROOT / ".env")

from task_1_sample_inference.inference import DashcamSchema, QwenVideoInferencer
from task_1_sample_inference.prompting import format_field_descriptions, load_prompt_template, render_prompt


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_VIDEO_DIR = "vlm_sanity_videos"
DEFAULT_REPORT_NAME = "vlm_inference_report.md"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"
PURPOSE_TEXT = "This script demonstrates an end-2-end inference pipeline + JSON parsing of the Qwen model."


@dataclass(frozen=True)
class InferenceResult:
    """Container for per-video inference results."""

    video_path: Path
    description: str
    attempts: int
    error: str | None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL inference over a folder of videos.")
    parser.add_argument("--videos-dir", default=DEFAULT_VIDEO_DIR, help="Folder with input videos.")
    parser.add_argument("--fps", type=float, required=True, help="Target sampling FPS.")
    parser.add_argument("--max-length", type=float, required=True, help="Max clip length in seconds.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--max-retries", type=int, default=1, help="Number of retries after a parse failure.")
    parser.add_argument("--out", default=None, help="Output markdown file path.")
    return parser


def _build_prompt(schema_model: type[DashcamSchema], prompt_path: Path) -> str:
    schema_json = json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
    template_text = load_prompt_template(prompt_path)
    return render_prompt(
        template_text,
        {
            "json_schema": schema_json,
            "field_descriptions": format_field_descriptions(schema_model),
        },
    )


def _collect_video_paths(videos_dir: Path, extension: str = ".mp4") -> list[Path]:
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    paths = [path for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() == extension]
    return sorted(paths, key=lambda path: path.name)


def _infer_videos(
    inferencer: QwenVideoInferencer,
    video_paths: list[Path],
    prompt: str,
    fps: float,
    max_frames: int,
    max_retries: int,
) -> list[InferenceResult]:
    results: list[InferenceResult] = []
    for video_path in video_paths:
        try:
            parsed = inferencer.infer(
                video_path=str(video_path),
                prompt=prompt,
                schema_model=DashcamSchema,
                sample_fps=fps,
                max_frames=max_frames,
                max_retries=max_retries,
            )
            description = parsed.get("description", "").strip()
            results.append(InferenceResult(video_path, description, max_retries + 1, None))
        except Exception as exc:
            results.append(InferenceResult(video_path, "", max_retries + 1, str(exc)))
    return results


def _render_markdown_report(
    results: list[InferenceResult],
    model_id: str,
    fps: float,
    max_length: float,
    max_frames: int,
    max_retries: int,
    output_path: Path,
    purpose: str,
) -> str:
    header = "\n".join(
        [
            "# VLM Inference Report",
            "",
            purpose,
            "",
            f"- Model: `{model_id}`",
            f"- Settings: fps={fps}, max_length={max_length}s, max_frames={max_frames}, max_retries={max_retries}",
            "",
            "| Video | VLM description |",
            "| --- | --- |",
        ]
    )
    rows: list[str] = []
    for result in results:
        relative_link = os.path.relpath(result.video_path, output_path.parent).replace(os.sep, "/")
        link = f"[{result.video_path.name}]({relative_link})"
        if result.error:
            description = f"ERROR after {result.attempts} attempts: {result.error}"
        else:
            description = result.description or "No description returned."
        rows.append(f"| {link} | {description} |")
    return "\n".join([header, *rows, ""])


def _resolve_output_path(output_arg: str | None, videos_dir: Path) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            return (REPO_ROOT / output_path).resolve()
        return output_path
    return videos_dir / DEFAULT_REPORT_NAME


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    max_frames = int(round(args.fps * args.max_length))
    if max_frames <= 0:
        raise ValueError("Computed max_frames must be positive.")

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_absolute():
        videos_dir = (REPO_ROOT / videos_dir).resolve()
    output_path = _resolve_output_path(args.out, videos_dir)
    video_paths = _collect_video_paths(videos_dir)
    if not video_paths:
        raise ValueError(f"No .mp4 videos found in: {videos_dir}")

    logger.info("Running inference on {} videos.", len(video_paths))
    inferencer = QwenVideoInferencer(
        model_id=args.model_id,
        device=args.device,
        dtype=torch.bfloat16,
    )
    prompt = _build_prompt(DashcamSchema, PROMPT_PATH)
    results = _infer_videos(
        inferencer=inferencer,
        video_paths=video_paths,
        prompt=prompt,
        fps=args.fps,
        max_frames=max_frames,
        max_retries=args.max_retries,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = _render_markdown_report(
        results=results,
        model_id=args.model_id,
        fps=args.fps,
        max_length=args.max_length,
        max_frames=max_frames,
        max_retries=args.max_retries,
        output_path=output_path,
        purpose=PURPOSE_TEXT,
    )
    output_path.write_text(report, encoding="utf-8")
    logger.info("Wrote markdown report to {}", output_path)


if __name__ == "__main__":
    main()

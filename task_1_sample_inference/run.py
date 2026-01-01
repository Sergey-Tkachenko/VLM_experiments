from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_1_sample_inference.inference import DashcamSchema, QwenVideoInferencer, load_video


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL video inference with JSON schema.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--fps", type=float, required=True, help="Target sampling FPS.")
    parser.add_argument(
        "--sampler",
        choices=["uniform", "windowed"],
        default="uniform",
        help="Sampling strategy.",
    )
    parser.add_argument("--start-sec", type=float, default=None, help="Window start in seconds (windowed only).")
    parser.add_argument("--window-sec", type=float, default=None, help="Window length in seconds (windowed only).")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--max-retries", type=int, default=1, help="Number of retries after a parse failure.")
    parser.add_argument("--out", default=None, help="Optional output JSON file. Defaults to stdout.")
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    args = _build_arg_parser().parse_args()
    sample = load_video(
        path=args.video,
        fps=args.fps,
        sampler=args.sampler,
        start_sec=args.start_sec,
        window_sec=args.window_sec,
    )
    logger.info("Effective FPS: {:.3f}", sample.effective_fps)

    inferencer = QwenVideoInferencer(
        model_id=args.model_id,
        device=args.device,
        dtype=torch.bfloat16,
    )
    result = inferencer.infer(
        frames=sample.frames,
        schema_model=DashcamSchema,
        max_retries=args.max_retries,
    )

    output_json = json.dumps(result, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output_json, encoding="utf-8")
    else:
        print(output_json)


if __name__ == "__main__":
    main()

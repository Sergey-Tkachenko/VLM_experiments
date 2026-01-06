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

# IMPORTANT: Load .env before importing modules that import `transformers` / `huggingface_hub`.
# These libraries may resolve cache locations at import time.
load_dotenv(REPO_ROOT / ".env")

from task_1_sample_inference.inference import DashcamSchema, QwenVideoInferencer
from task_1_sample_inference.prompting import format_field_descriptions, load_prompt_template, render_prompt


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL video inference with JSON schema.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--fps", type=float, required=True, help="Target sampling FPS.")
    parser.add_argument("--max-length", type=float, required=True, help="Max clip length in seconds.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--max-retries", type=int, default=1, help="Number of retries after a parse failure.")
    parser.add_argument("--out", default=None, help="Optional output JSON file. Defaults to stdout.")
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


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive.")
    max_frames = int(round(args.fps * args.max_length))
    if max_frames <= 0:
        raise ValueError("Computed max_frames must be positive.")
    logger.info("Max frames: {}", max_frames)

    inferencer = QwenVideoInferencer(
        model_id=args.model_id,
        device=args.device,
        dtype=torch.bfloat16,
    )
    prompt = _build_prompt(DashcamSchema, PROMPT_PATH)
    result = inferencer.infer(
        video_path=args.video,
        prompt=prompt,
        schema_model=DashcamSchema,
        sample_fps=args.fps,
        max_frames=max_frames,
        max_retries=args.max_retries,
    )

    output_json = json.dumps(result, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output_json, encoding="utf-8")
    else:
        print(output_json)


if __name__ == "__main__":
    main()

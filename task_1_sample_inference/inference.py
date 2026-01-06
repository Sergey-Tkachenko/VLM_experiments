from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from decord import VideoReader, cpu
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class DashcamSchema(BaseModel):
    """Schema for dashcam analysis outputs."""

    unique_pedestrian_count: int = Field(
        ...,
        description="Count of unique pedestrians visible in the clip. Use 0 if none are present.",
        ge=0,
    )
    adjacent_car_description: str = Field(
        ...,
        description="Short description of the car adjacent to the ego vehicle (color, type, position).",
        min_length=1,
    )


@dataclass(frozen=True)
class VideoSample:
    """Container for sampled video frames and realized FPS."""

    frames: np.ndarray
    effective_fps: float


def _compute_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    return max(1, int(round(source_fps / target_fps)))


def _ensure_even_frames(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] % 2 == 1 and frames.shape[0] > 1:
        return frames[:-1]
    return frames


def _extract_json_candidate(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = "\n".join(line for line in cleaned.splitlines() if "```" not in line).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def load_video(
    path: str,
    fps: float,
    sampler: Literal["uniform", "windowed"] = "uniform",
    start_sec: float | None = None,
    window_sec: float | None = None,
) -> VideoSample:
    """Load and sample a video using Decord.

    Args:
        path: Path to the video file.
        fps: Target sampling FPS.
        sampler: "uniform" or "windowed".
        start_sec: Window start in seconds for windowed sampling.
        window_sec: Window length in seconds for windowed sampling.

    Returns:
        VideoSample containing sampled frames and effective FPS.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = VideoReader(str(video_path), ctx=cpu(0))
    source_fps = float(reader.get_avg_fps())
    stride = _compute_stride(source_fps, fps)
    total_frames = len(reader)

    if sampler == "uniform":
        indices = np.arange(0, total_frames, stride)
    else:
        if start_sec is None or window_sec is None:
            raise ValueError("start_sec and window_sec are required for windowed sampling.")
        start_idx = int(start_sec * source_fps)
        end_idx = int((start_sec + window_sec) * source_fps)
        end_idx = min(end_idx, total_frames)
        if start_idx >= end_idx:
            raise ValueError("Windowed sampling range is empty. Check start_sec/window_sec.")
        indices = np.arange(start_idx, end_idx, stride)

    if indices.size == 0:
        raise ValueError("No frames selected. Adjust fps or window parameters.")

    try:
        frames = reader.get_batch(indices).asnumpy()
    except Exception:
        frames = np.stack([reader[int(idx)].asnumpy() for idx in indices])
    frames = _ensure_even_frames(frames)
    effective_fps = source_fps / stride
    return VideoSample(frames=frames, effective_fps=effective_fps)


def parse_to_json(text: str, schema_model: type[BaseModel]) -> dict[str, Any]:
    """Parse and validate JSON output using a Pydantic schema."""
    try:
        data = json.loads(_extract_json_candidate(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Model output is not valid JSON.") from exc

    try:
        return schema_model.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError("Model output failed schema validation.") from exc


class QwenVideoInferencer:
    """Inference helper for Qwen2.5-VL with schema-validated outputs."""

    @staticmethod
    def _to_video_uri(path: str) -> str:
        if path.startswith("file://"):
            return path
        return f"file://{Path(path).resolve()}"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model: Qwen2_5_VLForConditionalGeneration | None = None,
        processor: AutoProcessor | None = None,
    ) -> None:
        self.model = model or Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, attn_implementation="flash_attention_2")
        if model is None:
            self.model.to(device)
        self.processor = processor or AutoProcessor.from_pretrained(model_id)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

    def run_inference(
        self,
        video_path: str,
        prompt: str,
        sample_fps: float,
        max_frames: int | None,
        max_new_tokens: int = 256,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> str:
        video_message: dict[str, Any] = {
            "type": "video",
            "video": self._to_video_uri(video_path),
            "fps": sample_fps,
        }
        if max_frames is not None:
            video_message["max_frames"] = max_frames
        conversation = [
            {
                "role": "user",
                "content": [
                    video_message,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversation, return_video_kwargs=True
        )
        video_kwargs = dict(video_kwargs)
        video_kwargs.pop("fps", None)
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
            **({"max_pixels": max_pixels} if max_pixels is not None else {}),
            **({"min_pixels": min_pixels} if min_pixels is not None else {}),
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

    def run_batch_inference(
        self,
        video_paths: list[str],
        prompts: list[str],
        sample_fps: float,
        max_frames: int | None,
        max_new_tokens: int = 256,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> list[str]:
        if len(video_paths) != len(prompts):
            raise ValueError("video_paths and prompts must have the same length.")
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": self._to_video_uri(video_path),
                            "fps": sample_fps,
                            **({"max_frames": max_frames} if max_frames is not None else {}),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for video_path, prompt in zip(video_paths, prompts)
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conversation in conversations
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations, return_video_kwargs=True
        )
        video_kwargs = dict(video_kwargs)
        video_kwargs.pop("fps", None)
        inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
            **({"max_pixels": max_pixels} if max_pixels is not None else {}),
            **({"min_pixels": min_pixels} if min_pixels is not None else {}),
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def infer(
        self,
        video_path: str,
        prompt: str,
        schema_model: type[BaseModel],
        sample_fps: float,
        max_frames: int | None,
        max_retries: int = 1,
        max_new_tokens: int = 256,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> dict[str, Any]:
        attempts = max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                attempt_prompt = f"{prompt}\nYour previous response was invalid JSON. Respond with JSON only."
            else:
                attempt_prompt = prompt
            raw_text = self.run_inference(
                video_path,
                attempt_prompt,
                sample_fps=sample_fps,
                max_frames=max_frames,
                max_new_tokens=max_new_tokens,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
            )
            try:
                return parse_to_json(raw_text, schema_model)
            except ValueError as exc:
                last_error = exc
                logger.warning("Parsing failed on attempt {}: {}", attempt, exc)
        if last_error:
            raise last_error
        raise RuntimeError("Inference failed without producing output.")

    def infer_batch(
        self,
        video_paths: list[str],
        prompt: str,
        schema_model: type[BaseModel],
        sample_fps: float,
        max_frames: int | None,
        max_retries: int = 3,
        max_new_tokens: int = 256,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> list[InferenceOutcome]:
        """Run batched inference with a fixed prompt and retry JSON parsing failures per item."""
        if not video_paths:
            raise ValueError("video_paths batch must not be empty.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")

        outcomes: list[InferenceOutcome | None] = [None] * len(video_paths)
        pending = list(range(len(video_paths)))

        for attempt in range(1, max_retries + 2):
            if not pending:
                break
            if attempt > 1:
                # TODO: remove this retry prompt, always retry with the same prompt
                attempt_prompt = f"{prompt}\nYour previous response was invalid JSON. Respond with JSON only."
            else:
                attempt_prompt = prompt
            prompts = [attempt_prompt] * len(pending)
            batch_video_paths = [video_paths[idx] for idx in pending]
            outputs = self.run_batch_inference(
                batch_video_paths,
                prompts,
                sample_fps=sample_fps,
                max_frames=max_frames,
                max_new_tokens=max_new_tokens,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
            )
            next_pending: list[int] = []
            for idx, raw_text in zip(pending, outputs):
                try:
                    parsed = parse_to_json(raw_text, schema_model)
                    outcomes[idx] = InferenceOutcome(parsed=parsed, raw_text=raw_text, error=None, attempts=attempt)
                except ValueError as exc:
                    outcomes[idx] = InferenceOutcome(
                        parsed=None,
                        raw_text=raw_text,
                        error=str(exc),
                        attempts=attempt,
                    )
                    if attempt <= max_retries:
                        next_pending.append(idx)
            pending = next_pending

        return [
            outcome if outcome is not None else InferenceOutcome(None, "", "No output", 0)
            for outcome in outcomes
        ]


@dataclass(frozen=True)
class InferenceOutcome:
    """Container for batched inference outputs."""

    parsed: dict[str, Any] | None
    raw_text: str
    error: str | None
    attempts: int

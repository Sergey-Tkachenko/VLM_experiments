from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")


class ModelConfig(BaseModel):
    """Model settings for inference."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str
    device: str
    dtype: torch.dtype
    max_new_tokens: int

    @field_validator("dtype", mode="before")
    @classmethod
    def _validate_dtype(cls, value: Any) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if normalized in mapping:
                return mapping[normalized]
        raise ValueError(f"Unsupported dtype: {value}")

    @field_validator("max_new_tokens")
    @classmethod
    def _validate_max_new_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_new_tokens must be positive.")
        return value


class PreprocessConfig(BaseModel):
    """Configuration for preprocessing videos."""

    model_config = ConfigDict(extra="forbid")

    target_fps: float
    source_fps: float
    max_seconds: float
    max_pixels: int | None
    min_pixels: int | None

    @field_validator("target_fps", "source_fps", "max_seconds")
    @classmethod
    def _validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Value must be positive.")
        return value

    @field_validator("max_pixels", "min_pixels")
    @classmethod
    def _validate_optional_pixels(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("Pixel values must be positive.")
        return value


class HardwareConfig(BaseModel):
    """Configuration for hardware-dependent settings."""

    model_config = ConfigDict(extra="forbid")

    default_batch_size: int
    gpu_batch_sizes: dict[str, int]

    @field_validator("default_batch_size")
    @classmethod
    def _validate_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("default_batch_size must be positive.")
        return value


class EvalSettings(BaseModel):
    """Evaluation settings."""

    model_config = ConfigDict(extra="forbid")

    split: str
    split_path: Path
    dataset_root: Path
    xlsx_path: Path
    output_path: Path
    max_retries: int
    limit_batches: int | None
    dump_every_batches: int

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_retries must be >= 0.")
        return value

    @field_validator("dump_every_batches")
    @classmethod
    def _validate_dump_every(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dump_every_batches must be positive.")
        return value

    @field_validator("limit_batches")
    @classmethod
    def _validate_limit_batches(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("limit_batches must be positive when provided.")
        return value


class VisualizationParams(BaseModel):
    """Visualization settings for downstream tools."""

    model_config = ConfigDict(extra="forbid")

    point_half_window_sec: float = 0.5
    app_port: int = 5151

    @field_validator("point_half_window_sec")
    @classmethod
    def _validate_point_window(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("point_half_window_sec must be positive.")
        return value

    @field_validator("app_port")
    @classmethod
    def _validate_app_port(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("app_port must be positive.")
        return value


class EvalConfig(BaseModel):
    """Configuration for DADA evaluation."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    preprocess: PreprocessConfig
    hardware: HardwareConfig
    eval: EvalSettings
    visualization_params: VisualizationParams = VisualizationParams()


def load_eval_config(config_path: Path) -> EvalConfig:
    """Load and validate evaluation configuration from YAML.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed evaluation configuration.
    """
    load_dotenv(REPO_ROOT / ".env")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return EvalConfig.model_validate(raw)

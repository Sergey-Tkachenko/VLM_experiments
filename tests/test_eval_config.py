from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from task_2_evaluation_suite.config_loader import load_eval_config
from task_2_evaluation_suite.eval import resolve_batch_size


def test_resolve_batch_size_matches_gpu_name() -> None:
    batch_size = resolve_batch_size("NVIDIA RTX A4000", {"A4000": 3, "4090": 6}, default_batch=1)
    assert batch_size == 3


def test_load_eval_config_requires_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"id": "dummy"}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_eval_config(config_path)


def test_load_eval_config_defaults_visualization_params(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "id": "dummy",
                    "device": "cpu",
                    "dtype": "float32",
                    "max_new_tokens": 8,
                },
                "preprocess": {
                    "target_fps": 2.0,
                    "source_fps": 30.0,
                    "max_seconds": 4.0,
                    "max_pixels": None,
                    "min_pixels": None,
                },
                "hardware": {
                    "default_batch_size": 1,
                    "gpu_batch_sizes": {},
                },
                "eval": {
                    "split": "val",
                    "split_path": "val.json",
                    "dataset_root": "/tmp/dada",
                    "xlsx_path": "/tmp/dada.xlsx",
                    "output_path": "out.json",
                    "max_retries": 1,
                    "limit_batches": None,
                    "dump_every_batches": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_eval_config(config_path)
    assert config.visualization_params.point_half_window_sec == 0.5
    assert config.visualization_params.app_port == 5151

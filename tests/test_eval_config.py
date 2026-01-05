from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from task_2_evaluation_suite.eval import load_eval_config, resolve_batch_size


def test_resolve_batch_size_matches_gpu_name() -> None:
    batch_size = resolve_batch_size("NVIDIA RTX A4000", {"A4000": 3, "4090": 6}, default_batch=1)
    assert batch_size == 3


def test_load_eval_config_requires_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"id": "dummy"}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_eval_config(config_path)

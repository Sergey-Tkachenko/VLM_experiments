from pathlib import Path

import pytest

from task_2_evaluation_suite.viz_helpers import (
    build_accident_type_name,
    build_video_path,
    derive_dataset_name,
    parse_clip_key,
    resolve_accident_type_name,
)


def test_parse_clip_key_valid() -> None:
    assert parse_clip_key("1/010") == ("1", "010")


def test_parse_clip_key_invalid() -> None:
    with pytest.raises(ValueError):
        parse_clip_key("1-010")


def test_build_video_path() -> None:
    dataset_root = Path("/data")
    assert build_video_path(dataset_root, "2/123") == Path("/data/2/123/video.mp4")


def test_derive_dataset_name() -> None:
    assert derive_dataset_name(Path("artifacts/evals/dada_eval_val.json")) == "dada_eval_val"


def test_build_accident_type_name() -> None:
    entry = {"participants": ["ego_car", "car"], "interaction": "rear_end"}
    assert build_accident_type_name(entry) == "ego car and another car rear end"


def test_resolve_accident_type_name() -> None:
    mapping = {1: {"participants": ["ego_car"], "interaction": "crossing"}}
    assert resolve_accident_type_name(0, mapping) == "no accident"
    assert resolve_accident_type_name(1, mapping) == "ego car crossing"
    assert resolve_accident_type_name(99, mapping) == "unknown"

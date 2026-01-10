import json
from pathlib import Path

from openpyxl import Workbook

from task_1_sample_inference.inference import InferenceOutcome
from task_2_evaluation_suite.config_loader import EvalConfig
from task_2_evaluation_suite.eval import ClipMeta, _load_existing_results, _prepare_batch, run_eval


class DummyInferencer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def infer_batch(
        self,
        video_paths: list[str],
        prompt: str,
        schema_model: type,
        sample_fps: float,
        max_frames: int | None,
        max_retries: int = 3,
        max_new_tokens: int = 256,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> list[InferenceOutcome]:
        self.calls.append(video_paths)
        return [
            InferenceOutcome(
                parsed={"accident_type": 0, "accident_frame_position_sec": None},
                raw_text="{}",
                error=None,
                attempts=1,
            )
            for _ in video_paths
        ]


def _write_split(split_path: Path) -> None:
    split_path.write_text(json.dumps([[[1, 1], "clip-1"], [[2, 2], "clip-2"]]), encoding="utf-8")


def _write_xlsx(xlsx_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["Type", "Video", "Accident Frame"])
    sheet.append([1, 1, 30])
    sheet.append([2, 2, 60])
    workbook.save(xlsx_path)


def _build_config(
    split_path: Path,
    dataset_root: Path,
    xlsx_path: Path,
    output_path: Path,
) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "model": {
                "id": "dummy",
                "device": "cpu",
                "dtype": "float32",
                "max_new_tokens": 16,
            },
            "preprocess": {
                "target_fps": 2.0,
                "source_fps": 30.0,
                "max_seconds": 4.0,
                "max_pixels": None,
                "min_pixels": None,
            },
            "hardware": {
                "default_batch_size": 2,
                "gpu_batch_sizes": {},
            },
            "eval": {
                "split": "val",
                "split_path": split_path,
                "dataset_root": dataset_root,
                "xlsx_path": xlsx_path,
                "output_path": output_path,
                "max_retries": 1,
                "limit_batches": None,
                "dump_every_batches": 1,
            },
        }
    )


def test_prepare_batch_skips_missing_videos(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    existing_path = dataset_root / "1" / "001" / "video.mp4"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_bytes(b"")

    clips = [
        ClipMeta(type_id="1", video_id="001", clip_id="clip-1", accident_frame=None),
        ClipMeta(type_id="2", video_id="002", clip_id="clip-2", accident_frame=None),
    ]

    batch_clips, batch_paths = _prepare_batch(clips, [0, 1], dataset_root)

    assert len(batch_clips) == 1
    assert batch_clips[0].clip_id == "clip-1"
    assert batch_paths == [str(existing_path)]


def test_load_existing_results_ignores_invalid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    output_path.write_text("{", encoding="utf-8")

    assert _load_existing_results(output_path) == {}


def test_run_eval_resumes_and_skips_completed(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    dataset_root = tmp_path / "dataset"
    xlsx_path = tmp_path / "annotations.xlsx"
    output_path = tmp_path / "results.json"

    _write_split(split_path)
    _write_xlsx(xlsx_path)

    for type_id, video_id in [("1", "001"), ("2", "002")]:
        video_path = dataset_root / type_id / video_id / "video.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"")

    existing_results = {
        "1/001": {
            "clip_id": "clip-1",
            "prediction": {"accident_type": 5, "accident_frame_position_sec": 1.0},
            "json_valid": True,
            "error": None,
            "attempts": 1,
            "raw_text": "{}",
            "ground_truth": {},
        },
        "2/002": {
            "clip_id": "clip-2",
            "prediction": None,
            "json_valid": False,
            "error": "bad",
            "attempts": 1,
            "raw_text": "",
            "ground_truth": {},
        },
    }
    output_path.write_text(json.dumps(existing_results), encoding="utf-8")

    config = _build_config(split_path, dataset_root, xlsx_path, output_path)
    inferencer = DummyInferencer()
    results = run_eval(config, inferencer=inferencer, overwrite=False)

    assert len(inferencer.calls) == 1
    assert len(inferencer.calls[0]) == 1
    assert results["1/001"] == existing_results["1/001"]
    assert results["2/002"]["json_valid"] is True

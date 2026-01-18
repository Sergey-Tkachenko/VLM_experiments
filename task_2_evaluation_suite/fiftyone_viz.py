from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiftyone as fo
from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))

from task_2_evaluation_suite.dada_metadata import DadaMetadata, load_dada_metadata
from task_2_evaluation_suite.config_loader import load_eval_config
from task_2_evaluation_suite.viz_helpers import (
    build_video_path,
    derive_dataset_name,
    frame_to_seconds,
    load_type_mapping,
    normalize_float,
    normalize_int,
    parse_clip_key,
    resolve_accident_type_name,
)

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent / "type_id_to_parsed_description_mapping.json"
DEFAULT_ADDRESS = "0.0.0.0"

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class VizInputs:
    """Resolved inputs for FiftyOne visualization."""

    eval_json_path: Path
    dataset_root: Path
    xlsx_path: Path
    source_fps: float
    point_half_window_sec: float
    app_port: int
    mapping_path: Path
    dataset_name: str
    overwrite: bool
    remote: bool


def load_eval_results(eval_json_path: Path) -> dict[str, dict[str, Any]]:
    """Load evaluation results from JSON."""
    if not eval_json_path.exists():
        raise FileNotFoundError(f"Eval JSON not found: {eval_json_path}")
    raw = json.loads(eval_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Eval JSON must contain a dictionary of clip results.")
    return raw


def point_event(
    label: str,
    timestamp_sec: float,
    sample: fo.Sample,
    half_window_sec: float,
    conf: float | None = None,
    **attrs: Any,
) -> fo.TemporalDetection:
    """Represent a point timestamp as a short temporal segment."""
    start = max(0.0, timestamp_sec - half_window_sec)
    end = timestamp_sec + half_window_sec
    detection = fo.TemporalDetection.from_timestamps([start, end], sample=sample)
    detection.label = label
    if conf is not None:
        detection.confidence = float(conf)
    for key, value in attrs.items():
        detection[key] = value
    return detection


def _attach_metadata_fields(sample: fo.Sample, metadata: DadaMetadata | None, source_fps: float) -> None:
    if metadata is None:
        return
    sample["accident_frame"] = metadata.accident_frame
    sample["accident_frame_sec"] = frame_to_seconds(metadata.accident_frame, source_fps)
    sample["abnormal_start_frame"] = metadata.abnormal_start
    sample["abnormal_start_sec"] = frame_to_seconds(metadata.abnormal_start, source_fps)
    sample["abnormal_end_frame"] = metadata.abnormal_end
    sample["abnormal_end_sec"] = frame_to_seconds(metadata.abnormal_end, source_fps)
    sample["weather_id"] = metadata.weather_id
    sample["weather"] = metadata.weather
    sample["light_id"] = metadata.light_id
    sample["light"] = metadata.light
    sample["scene_id"] = metadata.scene_id
    sample["scene"] = metadata.scene
    sample["road_type_id"] = metadata.road_type_id
    sample["road_type"] = metadata.road_type
    sample["text"] = metadata.text
    sample["cause"] = metadata.cause
    sample["measure"] = metadata.measure


def _build_abnormal_window(
    sample: fo.Sample,
    metadata: DadaMetadata | None,
    source_fps: float,
) -> fo.TemporalDetections | None:
    if metadata is None:
        return None
    start_sec = frame_to_seconds(metadata.abnormal_start, source_fps)
    end_sec = frame_to_seconds(metadata.abnormal_end, source_fps)
    if start_sec is None or end_sec is None or end_sec < start_sec:
        return None
    window = fo.TemporalDetection.from_timestamps([start_sec, end_sec], sample=sample)
    window.label = "abnormal_window"
    return fo.TemporalDetections(detections=[window])


def _attach_prediction_fields(
    sample: fo.Sample,
    record: dict[str, Any],
    metadata: DadaMetadata | None,
    type_mapping: dict[int, dict[str, Any]],
    source_fps: float,
    point_half_window_sec: float,
) -> None:
    prediction = record.get("prediction") or {}
    ground_truth = record.get("ground_truth") or {}
    pred_type = normalize_int(prediction.get("accident_type"))
    gt_type = normalize_int(ground_truth.get("accident_type"))
    pred_time = normalize_float(prediction.get("accident_frame_position_sec"))
    gt_time = normalize_float(ground_truth.get("accident_frame_position_sec"))
    if gt_type is None:
        gt_type = normalize_int(sample["type_id"])
    if gt_time is None and metadata is not None:
        gt_time = frame_to_seconds(metadata.accident_frame, source_fps)

    sample["pred_accident_type"] = pred_type
    sample["gt_accident_type"] = gt_type
    sample["pred_accident_type_name"] = resolve_accident_type_name(pred_type, type_mapping)
    sample["gt_accident_type_name"] = resolve_accident_type_name(gt_type, type_mapping)
    sample["pred_accident_frame_sec"] = pred_time
    sample["gt_accident_frame_sec"] = gt_time

    pred_events = []
    gt_events = []
    if pred_time is not None:
        pred_events.append(
            point_event(
                label=f"pred:type={pred_type}",
                timestamp_sec=pred_time,
                sample=sample,
                half_window_sec=point_half_window_sec,
            )
        )
    if gt_time is not None:
        gt_events.append(
            point_event(
                label=f"gt:type={gt_type}",
                timestamp_sec=gt_time,
                sample=sample,
                half_window_sec=point_half_window_sec,
            )
        )

    if pred_events:
        sample["pred_event"] = fo.TemporalDetections(detections=pred_events)
    if gt_events:
        sample["gt_event"] = fo.TemporalDetections(detections=gt_events)

    sample["type_match"] = (pred_type == gt_type) if (pred_type is not None and gt_type is not None) else None
    if pred_time is not None and gt_time is not None:
        time_error = float(pred_time) - float(gt_time)
        sample["time_error_sec"] = time_error
        sample["abs_time_error_sec"] = abs(time_error)


def sget(sample: fo.Sample, field: str, default=None):
    # field may not exist in some datasets/versions
    if sample.has_field(field):
        v = sample.get_field(field)
        return default if v is None else v
    return default


def _attach_eval_fields(sample: fo.Sample, source_fps: float) -> None:
    # ---- classification labels ----
    pred_name = sget(sample, "pred_accident_type_name")
    gt_name   = sget(sample, "gt_accident_type_name")

    if pred_name is None:
        pid = sget(sample, "pred_accident_type")
        pred_name = str(pid) if pid is not None else None

    if gt_name is None:
        gid = sget(sample, "gt_accident_type")
        gt_name = str(gid) if gid is not None else None

    if pred_name is not None:
        sample["pred_type_cls"] = fo.Classification(label=pred_name)
    if gt_name is not None:
        sample["gt_type_cls"] = fo.Classification(label=gt_name)

    # ---- regression values (seconds) ----
    pred_t = sget(sample, "pred_accident_frame_sec")
    gt_t   = sget(sample, "gt_accident_frame_sec")

    if pred_t is not None:
        sample["pred_time_reg"] = fo.Regression(value=float(pred_t))
    if gt_t is not None:
        sample["gt_time_reg"] = fo.Regression(value=float(gt_t))

    # ---- convenience scalar: abs error in frames ----
    abs_err_sec = sget(sample, "abs_time_error_sec")
    if abs_err_sec is not None:
        sample["abs_time_error_frames"] = float(abs_err_sec) * float(source_fps)


def build_fiftyone_dataset(
    eval_results: dict[str, dict[str, Any]],
    dataset_root: Path,
    xlsx_path: Path,
    mapping_path: Path,
    source_fps: float,
    point_half_window_sec: float,
    dataset_name: str,
    overwrite: bool,
) -> fo.Dataset:
    """Build and populate a FiftyOne dataset for DADA results."""
    if fo.dataset_exists(dataset_name):
        if overwrite:
            fo.delete_dataset(dataset_name)
        else:
            logger.info("Dataset '{}' already exists; loading without rebuilding.", dataset_name)
            return fo.load_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    metadata_map = load_dada_metadata(xlsx_path)
    type_mapping = load_type_mapping(mapping_path)

    samples = []
    for clip_key, record in eval_results.items():
        try:
            type_id, video_id = parse_clip_key(clip_key)
        except ValueError as exc:
            logger.warning("Skipping clip_key {}: {}", clip_key, exc)
            continue
        video_path = build_video_path(dataset_root, clip_key)
        if not video_path.exists():
            logger.warning("Video missing for clip_key {}: {}", clip_key, video_path)
            continue
        sample = fo.Sample(filepath=str(video_path.resolve()))
        sample["clip_key"] = clip_key
        sample["type_id"] = type_id
        sample["video_id"] = video_id
        sample["clip_id"] = record.get("clip_id")
        sample["json_valid"] = bool(record.get("json_valid", False))
        sample["attempts"] = int(record.get("attempts") or 0)
        sample["error"] = record.get("error")
        sample["raw_text"] = record.get("raw_text")
        _attach_metadata_fields(sample, metadata_map.get((type_id, video_id)), source_fps)
        samples.append(sample)

    dataset.add_samples(samples)
    dataset.compute_metadata()

    for sample in dataset:
        clip_key = sample["clip_key"]
        record = eval_results.get(clip_key, {})
        metadata = metadata_map.get((sample["type_id"], sample["video_id"]))
        _attach_prediction_fields(
            sample,
            record,
            metadata,
            type_mapping,
            source_fps,
            point_half_window_sec,
        )
        _attach_eval_fields(sample, source_fps)
        abnormal_window = _build_abnormal_window(sample, metadata, source_fps)
        if abnormal_window is not None:
            sample["abnormal_window"] = abnormal_window
        sample.save()

    return dataset


def launch_fiftyone_app(
    dataset: fo.Dataset,
    port: int,
    remote: bool = True,
    address: str = DEFAULT_ADDRESS,
) -> fo.Session:
    """Launch the FiftyOne App for the given dataset."""
    return fo.launch_app(dataset, remote=remote, address=address, port=port)


def resolve_viz_inputs(
    eval_json_path: Path,
    config_path: Path,
    overwrite: bool,
    remote: bool,
) -> VizInputs:
    """Resolve visualization inputs from CLI args and optional config."""
    config = load_eval_config(config_path)
    resolved_dataset_root = config.eval.dataset_root
    resolved_xlsx_path = config.eval.xlsx_path
    resolved_source_fps = config.preprocess.source_fps
    resolved_point_window = config.visualization_params.point_half_window_sec
    resolved_app_port = config.visualization_params.app_port
    resolved_mapping_path = DEFAULT_MAPPING_PATH
    resolved_dataset_name = derive_dataset_name(eval_json_path)

    return VizInputs(
        eval_json_path=eval_json_path,
        dataset_root=resolved_dataset_root,
        xlsx_path=resolved_xlsx_path,
        source_fps=resolved_source_fps,
        point_half_window_sec=resolved_point_window,
        app_port=resolved_app_port,
        mapping_path=resolved_mapping_path,
        dataset_name=resolved_dataset_name,
        overwrite=overwrite,
        remote=remote,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize DADA-2000 eval results in FiftyOne.")
    parser.add_argument("--eval-json", type=Path, required=True, help="Path to eval JSON results.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "task_2_evaluation_suite/eval_config.yaml", help="Eval config YAML for defaults.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset.")
    parser.add_argument("--local", action="store_true", help="Launch the App locally instead of remote.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    inputs = resolve_viz_inputs(
        eval_json_path=args.eval_json,
        config_path=args.config,
        overwrite=args.overwrite,
        remote=not args.local,
    )
    eval_results = load_eval_results(inputs.eval_json_path)
    dataset = build_fiftyone_dataset(
        eval_results=eval_results,
        dataset_root=inputs.dataset_root,
        xlsx_path=inputs.xlsx_path,
        mapping_path=inputs.mapping_path,
        source_fps=inputs.source_fps,
        point_half_window_sec=inputs.point_half_window_sec,
        dataset_name=inputs.dataset_name,
        overwrite=inputs.overwrite,
    )
    session = launch_fiftyone_app(dataset, port=inputs.app_port, remote=inputs.remote, address=DEFAULT_ADDRESS)
    session.wait()


if __name__ == "__main__":
    main()

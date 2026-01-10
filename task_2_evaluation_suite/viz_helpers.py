from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task_2_evaluation_suite.dada_metadata import normalize_float, normalize_int


def parse_clip_key(clip_key: str) -> tuple[str, str]:
    """Parse a clip key like '1/010' into (type_id, video_id)."""
    parts = clip_key.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid clip key: {clip_key}")
    return parts[0], parts[1]


def build_video_path(dataset_root: Path, clip_key: str) -> Path:
    """Build the expected DADA video path for a clip."""
    type_id, video_id = parse_clip_key(clip_key)
    return dataset_root / type_id / video_id / "video.mp4"


def derive_dataset_name(eval_json_path: Path) -> str:
    """Derive a dataset name from the eval JSON filename."""
    return eval_json_path.stem


def format_participants(participants: list[str]) -> str:
    """Convert participant identifiers into a readable phrase."""
    readable = [participant.replace("_", " ") for participant in participants]
    if readable == ["ego car", "car"]:
        readable = ["ego car", "another car"]
    if len(readable) == 2:
        return f"{readable[0]} and {readable[1]}"
    if len(readable) == 1:
        return readable[0]
    return ", ".join(readable)


def build_accident_type_name(mapping_entry: dict[str, Any]) -> str:
    """Create a human-readable label from a mapping entry."""
    participants = mapping_entry.get("participants", [])
    interaction = mapping_entry.get("interaction", "unknown")
    participants_text = format_participants(participants)
    interaction_text = str(interaction).replace("_", " ")
    if participants_text:
        return f"{participants_text} {interaction_text}".strip()
    return interaction_text


def resolve_accident_type_name(type_id: int | None, mapping: dict[int, dict[str, Any]]) -> str | None:
    """Resolve a type ID into a readable label when possible."""
    if type_id is None:
        return None
    if type_id == 0:
        return "no accident"
    entry = mapping.get(type_id)
    if entry is None:
        return "unknown"
    return build_accident_type_name(entry)


def load_type_mapping(mapping_path: Path) -> dict[int, dict[str, Any]]:
    """Load the type ID mapping JSON."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {mapping_path}")
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in raw.items()}


def frame_to_seconds(frame: int | None, source_fps: float) -> float | None:
    """Convert a frame index to seconds."""
    if frame is None:
        return None
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    return float(frame) / source_fps

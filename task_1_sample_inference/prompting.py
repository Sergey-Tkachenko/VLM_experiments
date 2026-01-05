from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel


def load_prompt_template(template_path: Path) -> str:
    """Load a prompt template from disk."""
    return template_path.read_text(encoding="utf-8")


def format_field_descriptions(schema_model: type[BaseModel]) -> str:
    """Format schema field descriptions into a bullet list."""
    schema = schema_model.model_json_schema()
    properties = schema.get("properties", {})
    lines = []
    for name, meta in properties.items():
        field_type = meta.get("type", "unknown")
        desc = meta.get("description", "").strip()
        if desc:
            lines.append(f"- {name} ({field_type}): {desc}")
        else:
            lines.append(f"- {name} ({field_type})")
    return "\n".join(lines)


def render_prompt(template_text: str, context: dict[str, Any]) -> str:
    """Render a prompt template with the provided context."""
    env = Environment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(template_text)
    return template.render(**context)

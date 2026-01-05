from jinja2 import UndefinedError
import pytest

from task_1_sample_inference.inference import DashcamSchema
from task_1_sample_inference.prompting import format_field_descriptions, render_prompt


def test_format_field_descriptions_includes_fields() -> None:
    text = format_field_descriptions(DashcamSchema)
    assert "unique_pedestrian_count" in text
    assert "adjacent_car_description" in text


def test_render_prompt_substitutes_values() -> None:
    template = "Hello {{ name }}."
    rendered = render_prompt(template, {"name": "world"})
    assert rendered == "Hello world."

    with pytest.raises(UndefinedError):
        render_prompt(template, {})

from pathlib import Path

from task_2_evaluation_suite import eval as eval_module


def test_format_class_summaries_includes_expected_line() -> None:
    mapping_path = Path(eval_module.__file__).resolve().parent / "type_id_to_parsed_description_mapping.json"
    text = eval_module._format_class_summaries(mapping_path, [11])
    assert "11 -- participants -- ego car and another car, interaction -- hitting" in text

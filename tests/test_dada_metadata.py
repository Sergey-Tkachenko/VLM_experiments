from pathlib import Path

from openpyxl import Workbook

from task_2_evaluation_suite.dada_metadata import load_dada_metadata


def _write_test_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(
        [
            "type",
            "video",
            "accident frame",
            "abnormal start",
            "abnormal end",
            "weather(sunny,rainy,snowy,foggy)1-4",
            "light(day,night)1-2",
            "scenes(highway,tunnel,mountain,urban,rural)1-5",
            "linear(arterials,curve,intersection,T-junction,ramp) 1-5",
            "texts",
            "causes",
            "measures",
        ]
    )
    sheet.append([1, 7, 120, 90, 150, 2, 1, 4, 3, "text", "cause", "measure"])
    workbook.save(path)


def test_load_dada_metadata_parses_fields(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "dada.xlsx"
    _write_test_workbook(xlsx_path)

    metadata = load_dada_metadata(xlsx_path)
    entry = metadata[("1", "007")]
    assert entry.accident_frame == 120
    assert entry.abnormal_start == 90
    assert entry.abnormal_end == 150
    assert entry.weather == "rainy"
    assert entry.light == "day"
    assert entry.scene == "urban"
    assert entry.road_type == "intersection"
    assert entry.text == "text"
    assert entry.cause == "cause"
    assert entry.measure == "measure"

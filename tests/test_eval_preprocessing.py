from task_2_evaluation_suite.eval import compute_max_frames


def test_compute_max_frames() -> None:
    assert compute_max_frames(target_fps=2.0, max_seconds=3.0) == 6

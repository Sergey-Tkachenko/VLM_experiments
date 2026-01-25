from task_2_evaluation_suite.eval import compute_max_frames_for_window


def test_compute_max_frames_for_window() -> None:
    assert compute_max_frames_for_window(target_fps=2.0, pre_buffer_sec=2.0, post_buffer_sec=1.0) == 6

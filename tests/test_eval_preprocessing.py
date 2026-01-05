import numpy as np

from task_2_evaluation_suite.eval import sample_and_clip_frames


def test_sample_and_clip_frames_pads_repeat_first() -> None:
    frames = [np.full((2, 2, 3), idx, dtype=np.uint8) for idx in range(10)]
    sampled = sample_and_clip_frames(
        frames=frames,
        source_fps=10.0,
        target_fps=2.0,
        max_seconds=3.0,
        padding="repeat_first",
    )
    assert sampled.shape == (6, 2, 2, 3)
    assert sampled[0, 0, 0, 0] == 0
    assert sampled[1, 0, 0, 0] == 5
    assert sampled[-1, 0, 0, 0] == 0

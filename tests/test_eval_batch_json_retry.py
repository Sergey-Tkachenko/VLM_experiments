import numpy as np

from task_1_sample_inference.inference import InferenceOutcome, QwenVideoInferencer
from task_2_evaluation_suite.eval import DadaAccidentSchema


class DummyInferencer(QwenVideoInferencer):
    def __init__(self, outputs_per_call: list[list[str]]) -> None:
        self.outputs_per_call = outputs_per_call
        self.call_idx = 0

    def run_batch_inference(  # type: ignore[override]
        self,
        videos: list[np.ndarray],
        prompts: list[str],
        sample_fps: float | list[float],
        max_new_tokens: int = 256,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> list[str]:
        outputs = self.outputs_per_call[self.call_idx]
        self.call_idx += 1
        return outputs


def test_infer_batch_retries_failed_json() -> None:
    outputs = [
        ['{"accident_type": 1, "accident_frame_position_sec": 0.5}', "{bad json}"],
        ['{"accident_type": 2, "accident_frame_position_sec": 1.0}'],
    ]
    inferencer = DummyInferencer(outputs)
    videos = [np.zeros((2, 2, 2, 3), dtype=np.uint8) for _ in range(2)]
    results = inferencer.infer_batch(
        videos,
        "test prompt",
        DadaAccidentSchema,
        sample_fps=2.0,
        max_retries=1,
    )

    assert isinstance(results[0], InferenceOutcome)
    assert results[0].parsed is not None
    assert results[0].attempts == 1
    assert results[1].parsed is not None
    assert results[1].attempts == 2

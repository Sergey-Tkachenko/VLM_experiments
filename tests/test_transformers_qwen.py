import os

import pytest
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig


@pytest.mark.integration
def test_qwen_config_available() -> None:
    """Validate that Qwen2.5-VL config is importable and registered."""
    config = Qwen2_5_VLConfig()
    assert config.model_type == "qwen2_5_vl"
    assert AutoConfig.for_model("qwen2_5_vl").model_type == "qwen2_5_vl"


@pytest.mark.integration
def test_optional_model_download() -> None:
    """Optionally download Qwen2.5-VL artifacts to validate HF connectivity."""
    if os.environ.get("VLM_TEST_MODEL_DOWNLOAD") != "1":
        return

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    assert processor is not None

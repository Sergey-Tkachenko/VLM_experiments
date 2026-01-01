import os

import torch


def test_cuda_smoke() -> None:
    """Run a minimal CUDA operation to validate the runtime and driver."""
    if os.environ.get("VLM_SKIP_CUDA_TEST") == "1":
        return

    assert torch.cuda.is_available(), "CUDA is not available; check drivers or GPU assignment."
    device = torch.device("cuda")
    tensor = torch.randn((256, 256), device=device)
    result = torch.mm(tensor, tensor)
    assert result.is_cuda
    assert torch.isfinite(result).all().item()

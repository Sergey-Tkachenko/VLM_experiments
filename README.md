# VLM_experiments

## Setup

This repo uses `uv` for dependency management. The RunPod image already includes PyTorch 2.4, so we avoid reinstalling it
by creating a venv that can access system site packages.

```bash
python -m pip install uv
uv venv --system-site-packages
uv sync --group dev
```

Optional: install `ffmpeg` system-wide if you want video IO tests to exercise codec support.

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Tests

Run the full test suite:

```bash
uv run pytest -v
```

Environment toggles:

- `VLM_SKIP_CUDA_TEST=1` skips the CUDA smoke test.
- `VLM_SKIP_DECORD_TEST=1` skips the decord + ffmpeg video IO test.
- `VLM_TEST_MODEL_DOWNLOAD=1` enables a real HF download test for Qwen2.5-VL.

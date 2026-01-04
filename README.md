# VLM_experiments

## Citation

If you use the DADA-2000 dataset, please cite:

Fang, Jianwu, Dingxin Yan, Jiahuan Qiao, Jianru Xue, and Hongkai Yu. "DADA: Driver Attention Prediction in Driving Accident Scenarios." IEEE Transactions on Intelligent Transportation Systems 23, no. 6 (2022): 4959-4971.

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

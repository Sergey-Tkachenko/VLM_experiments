# VLM_experiments

## Citation

If you use the DADA-2000 dataset, please cite:

Fang, Jianwu, Dingxin Yan, Jiahuan Qiao, Jianru Xue, and Hongkai Yu. "DADA: Driver Attention Prediction in Driving Accident Scenarios." IEEE Transactions on Intelligent Transportation Systems 23, no. 6 (2022): 4959-4971.

## Setup

This repo uses `uv` for dependency management. The venv installs PyTorch locally so it does not rely on system site packages.

```bash
python -m pip install uv
uv venv
uv sync --group dev
```

Optional: install `ffmpeg` system-wide if you want video IO tests to exercise codec support.

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## DADA-2000 splits

The DADA-2000 evaluation splits are rebuilt with these main decisions:

- Keep only the 19 popular `type` labels: 11, 43, 50, 10, 5, 6, 37, 48, 38, 8, 1, 57, 12, 49, 56, 14, 39, 42, 9.
- Honor the provided origin `train/val/test` assignments for those labels, dropping unpopular classes from the existing splits.
- Assign remaining clips using per-class proportional stratification with largest-remainder rounding and a fixed seed.
- Use target ratio weights `3/1/3` for train/val/test, keeping class proportions consistent across splits.
- If a clip appears in multiple splits, log a warning and default it to `test`.

## Tests

Run the full test suite:

```bash
uv run pytest -v
```

Environment toggles:

- `VLM_SKIP_CUDA_TEST=1` skips the CUDA smoke test.
- `VLM_SKIP_DECORD_TEST=1` skips the decord + ffmpeg video IO test.
- `VLM_TEST_MODEL_DOWNLOAD=1` enables a real HF download test for Qwen2.5-VL.

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


## Getting assets (Mixkit)

This project uses a small set of short stock videos hosted on Mixkit.

### Videos (Mixkit item pages)
- Highway from the point of view of a car (0:08): https://mixkit.co/free-stock-video/highway-from-the-point-of-view-of-a-car-21571/
- Side view of a vehicle in traffic, time lapse shot (0:09): https://mixkit.co/free-stock-video/side-view-of-a-vehicle-in-traffic-time-lapse-shot-42036/
- Time lapse of traffic at night: https://mixkit.co/free-stock-video/time-lapse-of-traffic-at-night-4240/
- Top view of the traffic around a roundabout at night: https://mixkit.co/free-stock-video/top-view-of-the-traffic-around-a-roundabout-at-night-50998/
- Two avenues with many cars traveling at night: https://mixkit.co/free-stock-video/two-avenues-with-many-cars-traveling-at-night-34562/
- Curving a road while cars speeding by: https://mixkit.co/free-stock-video/curving-a-road-while-cars-speeding-by-34564/

### License + compliance notes
- Each linked item is labeled on Mixkit as downloadable “under the Mixkit Stock Video Free License”.
- Mixkit states that Free License videos are free to download and use in commercial projects, and no attribution is required (though appreciated).
- This repo does **not** redistribute any Mixkit video files. It only provides links and a script to download from Mixkit.
- Don’t use any downloader to “mass download” Mixkit’s library; keep usage limited to the small list above.

References:
- Mixkit License: https://mixkit.co/license/
- Mixkit Terms: https://mixkit.co/terms/

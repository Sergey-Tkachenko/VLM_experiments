# Be accurate with aligning time

In the first task, we pass frames directly as a sequence of images. However, the default load function drops information about absolute time position. While not critical for demo purposes, we need to account for it in real tasks, because Qwen relies on absolute time ids in it MRoPE layers.


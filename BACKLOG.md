# Memory and timelines

It is unclear how much memory we do need for KV, weights and actually data/activations. Same goes for time -- we do not know hot to cut timings.

That's,  I need to build calculator for Qwen 2.5 VL to understand its limits. It should accept model and GPU specs, and output necessary metrics: TTFT, tokens/sec, memory consumption for different context length and so on.

# Inference optimization

I completed the task_1 yet script takes long to run. Partially, this is because we load weights every time.

However, I am not sure even basic inference optiomizations are applied -- e.g. KV cache.

# Another inference engine

HF transformers have their own limitations, e.g. they lack structured output feature. That is why we may need to think about different engine, e.g. vLLM.


# Research QWEN configuration

Qwen 2.5 VL has a lot of configs, which I did not touch in the task 1. They might offer space of optimization, that is, we need to investigate it further.

# Expand qualititative analysis further

20 -> 100, goal -- ensure noise level is acceptable (< 5%). Make sure you respect train/test/val split, regarding only train/val samples.

## 2-week VLM-for-dashcam “read + build” one-pager (checklist)

### Track A — **Baseline pipeline** (Video → VLM → strict JSON) ✅

**Read (only what you’ll actually use in code):**

* **Transformers Qwen2.5-VL doc**: how to pass **video inputs** (`{"type":"video","path":...}`) ([Hugging Face][1])
* **Decord** (fast random access + `get_batch`) *or* **PyAV** (FFmpeg bindings when codecs get spicy) ([GitHub][2])
* **Qwen2.5-VL report**: what “long-video comprehension / temporal grounding” means for this family ([arXiv][3])

**Build deliverables:**

* [ ] `load_video(path) -> frames[t]` with **two samplers**: uniform + windowed
* [ ] `infer(frames, prompt) -> raw_text`
* [ ] `parse_to_json(raw_text) -> dict` + **invalid-json rate** metric
* [ ] CLI: `python run.py --video ... --mode dashcam|cabin --out result.json`

---

### Track B — **Eval harness** (2 tasks, reproducible) ✅

Pick **2 of 3** tasks:

1. **DADA-2000** = accident/risk categories + accident window (dashcam) ([arXiv][4])
2. **Drive&Act** = fine-grained in-cabin driver behavior (cabin) ([CVF Open Access][5])
3. **BDD-X** = action + textual explanation (report-style outputs) ([GitHub][6])

**Read:**

* **VLMEvalKit** (mindset + harness patterns; even if you don’t use it directly, copy the approach) ([GitHub][7])
* Dataset paper(s) for whichever two you choose (DADA-2000 / Drive&Act / BDD-X) ([arXiv][4])
* **FiftyOne** (fast visual sanity-checking of clips/labels; huge time saver) ([GitHub][8])

**Build deliverables:**

* [ ] Dataset adapters: `dataset[i] -> {video_path, label, extra_meta}`
* [ ] Prompt templates per task (zero-shot + 1 “structured JSON” variant)
* [ ] Metrics: macro-F1 (classification) / exact-match + parsing success (structured)
* [ ] `eval.py` outputs a single CSV: model, sampler, prompt_id, score, json_fail_rate

---

### Track C — **Fine-tuning (QLoRA) that beats your baseline** ✅

**Read (Tier 0 for your goal):**

* **LoRA** (what adapters are, where they attach) ([arXiv][9])
* **QLoRA** (4-bit + adapters; why this fits your <$100 constraint) ([arXiv][10])
* **HF PEFT docs** (the exact knobs you’ll use) ([Hugging Face][11])

**Build deliverables:**

* [ ] Choose **one** target first: Drive&Act *coarse* distraction OR DADA-2000 *coarse* accident category
* [ ] Train QLoRA adapters (small run → real run)
* [ ] Before/after table vs your prompted baseline (same eval harness)
* [ ] Export: `adapter/` + `repro.md` (commands + key hyperparams)

---

### Track D — **Production-shaped constraints (throughput + “near-online” early exit)** ✅

**Read:**

* **Qwen3-VL report** (newest stack; long multimodal context + stronger video dynamics; use as “where the ecosystem is going”) ([arXiv][12])
* **vLLM VLM docs** (only to internalize batching/serving constraints; support is “experimental”) ([vLLM][13])

**Build deliverables:**

* [ ] Throughput script: decode_time + infer_time vs #frames; extrapolate to 1k/10k clips/day
* [ ] Early-exit prototype: run in windows until confidence > τ; plot accuracy vs latency proxy

---

## Reading list — rewritten tiers (tied to artifacts)

### Tier 0 (must for *shipping-ready* in 2 weeks)

* Qwen2.5-VL report ([arXiv][3])
* Transformers Qwen2.5-VL (video input usage) ([Hugging Face][1])
* LoRA ([arXiv][9])
* QLoRA ([arXiv][10])
* HF PEFT docs ([Hugging Face][11])
* VLMEvalKit ([GitHub][7])
* Decord or PyAV ([GitHub][2])
* 2 dataset docs (pick 2): DADA-2000 / Drive&Act / BDD-X ([arXiv][4])

### Tier 1 (high value, but only if Tier 0 is done)

* Qwen-VL (original design + training stages) ([arXiv][14])
* Qwen2-VL (dynamic resolution + M-RoPE; image+video unified paradigm) ([arXiv][15])
* LLaVA (instruction tuning archetype) ([arXiv][16])
* Video-LLaVA (video VLM framing + alignment-before-projection idea) ([arXiv][17])
* BLIP-2 (bridge pattern intuition) ([arXiv][18])

### Tier 2 (optional / later)

* Qwen3-VL report (read early only if your team is already migrating fast) ([arXiv][12])
* LLaVA-Video (synthetic video instruction tuning idea) ([arXiv][19])

If you tell me which **two** tasks you’re picking (DADA vs Drive&Act vs BDD-X), I’ll collapse this further into a *single* exact reading order + the exact artifacts you should produce by day 3/7/14.

[1]: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl?utm_source=chatgpt.com "Qwen2.5-VL"
[2]: https://github.com/dmlc/decord?utm_source=chatgpt.com "dmlc/decord: An efficient video loader for deep learning ..."
[3]: https://arxiv.org/abs/2502.13923?utm_source=chatgpt.com "Qwen2.5-VL Technical Report"
[4]: https://arxiv.org/abs/1904.12634?utm_source=chatgpt.com "DADA-2000: Can Driving Accident be Predicted by Driver Attention? Analyzed by A Benchmark"
[5]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Martin_DriveAct_A_Multi-Modal_Dataset_for_Fine-Grained_Driver_Behavior_Recognition_in_ICCV_2019_paper.pdf?utm_source=chatgpt.com "Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver ..."
[6]: https://github.com/JinkyuKimUCB/BDD-X-dataset?utm_source=chatgpt.com "JinkyuKimUCB/BDD-X-dataset: Berkeley Deep Drive- ..."
[7]: https://github.com/open-compass/VLMEvalKit?utm_source=chatgpt.com "open-compass/VLMEvalKit"
[8]: https://github.com/voxel51/fiftyone?utm_source=chatgpt.com "voxel51/fiftyone: Refine high-quality datasets and visual AI ..."
[9]: https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"
[10]: https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com "QLoRA: Efficient Finetuning of Quantized LLMs"
[11]: https://huggingface.co/docs/transformers/en/peft?utm_source=chatgpt.com "PEFT"
[12]: https://arxiv.org/abs/2511.21631?utm_source=chatgpt.com "Qwen3-VL Technical Report"
[13]: https://docs.vllm.ai/en/v0.6.1/models/vlm.html?utm_source=chatgpt.com "Using VLMs - vLLM"
[14]: https://arxiv.org/abs/2308.12966?utm_source=chatgpt.com "Qwen-VL: A Versatile Vision-Language Model for ..."
[15]: https://arxiv.org/abs/2409.12191?utm_source=chatgpt.com "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
[16]: https://arxiv.org/abs/2304.08485?utm_source=chatgpt.com "Visual Instruction Tuning"
[17]: https://arxiv.org/html/2311.10122v3?utm_source=chatgpt.com "Video-LLaVA: Learning United Visual Representation by ..."
[18]: https://arxiv.org/abs/2301.12597?utm_source=chatgpt.com "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
[19]: https://arxiv.org/abs/2410.02713?utm_source=chatgpt.com "Video Instruction Tuning With Synthetic Data"

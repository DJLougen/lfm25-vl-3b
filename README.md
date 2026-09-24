# LFM2.5-VL-3B and DSpark on MLX (Apple Silicon)

## What

This project converts Liquid AI's LFM2.5-VL-3B-DSpark speculative drafter from
its upstream SGLang checkpoint into MLX-native BF16, affine 8-bit, and affine
4-bit variants, then measures them with MLX-VLM targets. It also records target
precision fidelity and a separate, unsuccessful Core ML/ANE experiment.

This repository contains source, raw measurements, and provenance. It does not
contain the target checkpoints, draft weights, downloaded COCO photos, or
generated Core ML packages. Get the three published draft variants here:

- [BF16 drafter](https://huggingface.co/DJLougen/LFM2.5-VL-3B-DSpark-MLX-bf16)
- [8-bit drafter](https://huggingface.co/DJLougen/LFM2.5-VL-3B-DSpark-MLX-8bit)
- [4-bit drafter](https://huggingface.co/DJLougen/LFM2.5-VL-3B-DSpark-MLX-4bit)

The target models remain upstream:
[BF16](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-MLX-bf16) and
[8-bit](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-MLX-8bit).

## Why

Liquid AI's DSpark draft checkpoint is SGLang-oriented. MLX-VLM can use a
compatible DFlash drafter with the LFM2.5-VL-3B target, but the weight layout
and runtime configuration need conversion. This repository makes that
conversion reproducible and checks the real Apple-Silicon execution path
rather than assuming that draft acceptance alone means a speedup.

## Why it matters

Speculative decoding proposes several tokens and asks the target model to
verify them. When used greedily, the target's verified tokens remain the
output; the drafter only helps if verification costs less than the target-only
decode it replaces. The tested results show a useful short-prompt gain with
the BF16 target, only a small mean gain with the requested 8-bit target, and
slower decoding on long contexts. That distinction determines when to enable
the draft model.

## Results

All measurements below were made on one Apple M3 Max with 36 GB unified
memory, Python 3.12, `mlx==0.32.2`, `mlx-vlm==0.7.3`, and
`transformers==5.17.0`. They are local measurements, not Liquid AI's published
M5 Max benchmark.

### Short-prompt precision screen

Three instructions are run against the same image, with greedy decoding and
two repetitions per prompt/configuration. The runner retains the repetition
with the lowest end-to-end wall time for each prompt/configuration, then the
summary averages across prompts. This best-of-two selection can be optimistic;
the screen is exploratory. `results/sweep.json` stores the selected per-prompt
rows, not both repetitions; `results/stageA.jsonl` contains the separate
two-repetition context subset.

| Target | Drafter | Block | Decode tok/s | Decode ratio | End-to-end ratio | Accepted/round |
|---|---|---:|---:|---:|---:|---:|
| BF16 | target only | - | 43.2 / 41.9 / 45.8 | 1.00x | 1.00x | - |
| BF16 | DSpark BF16 | 8 | 142.6 / 110.3 / 139.1 | 2.99x | 2.41x | 4.30 |
| BF16 | DSpark 8-bit | 8 | 141.8 / 108.7 / 145.3 | 3.02x | 2.44x | 4.30 |
| BF16 | DSpark 4-bit | 8 | 133.6 / 104.6 / 127.0 | 2.79x | 2.31x | 4.09 |
| 8-bit | target only | - | 69.0 / 78.1 / 77.7 | 1.00x | 1.00x | - |
| 8-bit | DSpark 8-bit | 4 | 93.5 / 74.2 / 78.9 | 1.11x | 1.09x | 2.99 |
| 8-bit | DSpark 4-bit | 4 | 93.0 / 70.2 / 76.1 | 1.08x | 1.06x | 2.92 |

The selected draft output for each prompt matched its target-only text. With
an 8-bit target, block 8 was slower than target-only for all three drafter
precisions. The small block-4 mean gain is workload-specific; use target-only
by default unless a local measurement confirms the drafter helps.

### Long-context sweep

Paired sweep with a BF16 target and BF16 DSpark drafter, one image, the same
document-and-image prompt, greedy decoding, 128-token cap, and two serial runs
per configuration. “Book context” is the requested number of book tokens
added to the prompt; total input also includes image and template tokens.
Values are median generated tokens/s. Exact raw rows are in
`results/stageB_fullctx.jsonl`.

| Book context | Target only | Draft block 4 | Draft block 6 | Draft block 8 | Result |
|---:|---:|---:|---:|---:|---|
| 0 | 49.05 | 128.00 | 181.61 | 198.19 | Block 8; about 4.0x |
| 8,192 | 47.61 | 66.83 | 81.12 | 78.14 | Block 6; about 1.7x |
| 16,384 | 26.45* | 35.34* | 63.36 | 57.10 | No stable recommendation |
| 32,768 | 6.21 | 5.46 | 4.69 | 4.37 | Target only |
| 65,536 | 6.45 | 3.90 | 3.40 | 3.17 | Target only |

All 40 runs emitted the same greedy text hash as target-only for their
context. At 32k and 64k, however, drafting reduced rather than increased
decode throughput. Acceptance fell to about 1.8 tokens/round at 32k and 1.5
at 64k. **Do not enable DSpark for 32k+ prompts on the basis of these
measurements.** At 16k, the target-only and block-4 medians are dominated by
an unusually slow second run and are not stable speed claims.

### Target fidelity and other screens

`results/fidelity.json` compares target BF16 reference logits with FP16 and
8-bit target variants over 1,745 response positions from seven independent
clusters. This is an exploratory sample, not an equivalence certification.
Paired cluster-bootstrap 95% intervals for delta NLL include zero, but top-1
agreement is lower for both variants. Only 1/10 FP16 and 2/10 8-bit greedy
outputs were wholly identical to the BF16 output. Both variants scored 5/5
chart values and zero document-transcription character error on this small
task set; those task counts are not a general quality guarantee.

Other closed screens:

- One 64k run with an 8-bit target KV cache was slower to first token
  (206 s versus 90–91 s in the two BF16-cache controls), had lower measured
  prefill throughput (320 versus 723 tokens/s), used more peak memory (16.6
  versus 9.0 GB), and produced a different text hash. This is not a quality
  evaluation and gives no reason to enable the tested KV-cache quantization.
- A 64k run with an 8-token DSpark block and 4096-token draft window matched
  target-only text, but was slower (4.93 versus 5.99/6.91 tok/s), with 1.89
  accepted tokens/round. The window did not rescue long-context drafting.
- The optional Core ML/ANE layers-0–1 E1 prototype is **NO-GO** for numerical
  parity and speed. See `ane/E1_REPORT.md`; it is not part of the runtime.

## How to use

The pinned environment is for Apple Silicon and Python 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the upstream 8-bit target and the 8-bit drafter into ignored local
directories:

```bash
hf download LiquidAI/LFM2.5-VL-3B-MLX-8bit \
  --local-dir LFM2.5-VL-3B-MLX-8bit
hf download DJLougen/LFM2.5-VL-3B-DSpark-MLX-8bit \
  --local-dir LFM2.5-VL-3B-DSpark-MLX-8bit
```

With the 8-bit target, the measured best drafter setting was block 4, though
the mean gain was small:

```bash
.venv/bin/python -m mlx_vlm.generate \
  --model LFM2.5-VL-3B-MLX-8bit \
  --draft-model LFM2.5-VL-3B-DSpark-MLX-8bit \
  --draft-block-size 4 --temperature 0 \
  --image image.jpg --prompt "Describe this image." --max-tokens 128
```

For a BF16 target, block 8 was best in the short-prompt screen. DSpark in the
tested MLX-VLM version is greedy-only; use `--temperature 0`. Compare against
target-only decoding on the prompts you actually use. See `AGENTS.md` for the
script map and `PROVENANCE.json` for model revisions, hashes, gates, and
artifact records.

To reproduce the benchmark, `make_workload.py` reconstructs the local
workload, downloading COCO images from the URLs recorded in
`data/workload.json`; photos are not committed. The full-context sweep command
and exploratory fidelity command are recorded in `PROVENANCE.json`. The
fidelity script requires the separate `quantization-evidence` qelab package,
which is not included in this public repository.

## Known issues

- DFlash speculation was greedy-only with the tested MLX-VLM version.
- The 8-bit-target gains were small on a three-prompt screen and may disappear
  on other prompts or devices.
- The BF16-drafter long-context sweep showed drafting slower at 32k and 64k;
  performance of quantized drafters at those contexts was not established.
- The 16k speed measurements had large run-to-run variance.
- Target FP16/8-bit weights are lossy variants; see the small-sample caveats in
  `results/fidelity.json`.
- The source repo's code is Apache-2.0. Published draft weights are derived
  model artifacts under Liquid AI's LFM Open License v1.0; the code license
  does not apply to the weights. See `NOTICE` and the model cards.
- No release version tag is created; this publication does not claim a
  versioned software release.

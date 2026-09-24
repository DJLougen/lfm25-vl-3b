# E1 Report — ANE prefill for LFM2.5-VL-3B layers 0–1 (go/no-go gate)

**Date:** 2026-09-24 · **Machine:** M3 Max (30-core GPU, 36 GB unified) · **Verdict: NO-GO**

## What was built

`export_lfm2_conv.py` ports decoder layers 0–1 of LFM2.5-VL-3B's text backbone
(both are ShortConv layers — verified from checkpoint keys: layers 0/1 have
`conv.*` weights, layer 2 does not) to the layafast ANE layout
(`laya-mlx/ane/ane_model.py` pattern, Apache-2.0, ported with attribution):

- activations in BC1S form `[1, 2048, 1, C]`, all linears as 1×1 Conv2d
- depthwise causal conv k=3 as `Conv2d(2048, 2048, (1,3), groups=2048)` over
  `concat([state(2), Bx])` along the sequence axis, VALID padding
- state convention identical to the MLX ShortConv cache: last 2 positions of
  `B*x` (pre-conv), exported as an output, re-fed as input
- RMSNorm: **fp32 reduction island** — activations stay fp16, mean-of-squares
  is computed in fp32 (fp16 squares near the 65504 max would overflow),
  `rsqrt` result cast back to fp16 before the affine. A standalone probe
  confirmed the ANE keeps this fp32 reduction to rel-err ~1e-4.
- FFN dim **10752** (read from `feed_forward.w1.weight [10752, 2048]`).

Exported via `ct.convert` (mlprogram, FLOAT16, `CPU_AND_NE`, macOS15 target)
for C=128, 256, 512 in the laya-mlx venv (coremltools 9.0, torch 2.14).

## Placement (MLComputePlan)

All **54 compute ops** prefer `MLNeuralEngineComputeDevice` at every chunk
size; the only non-ANE ops are 106 `const` nodes (folded weights, zero
estimated cost; estimated cost is 100% ANE). Note: laya's `export.py` device
filter is buggy — it tests for device names `"ANE"`/`"NeuralEngine"` but the
real class name is `MLNeuralEngineComputeDevice`, so its "non_ane_ops" report
mislabels every ANE op as unknown. Fixed here.

## Numerics — **FAIL** (gate: cosine_min ≥ 0.9995)

Reference: MLX fp16 modules from `mlx_vlm.models.lfm2` in the lfm25 venv,
real embeddings of the first 1024 tokens of `data/book.txt` (Pride and
Prejudice), layers 0–1 with a live conv cache, saved to `ref_outputs.npz`.

| check | C=128 | C=512 |
|---|---|---|
| cosine_min | 0.9906 | 0.9884 |
| cosine_mean | 0.9947 | 0.9943 |
| max abs err | 0.0337 | 0.0386 |
| state0 cos_min / maxabs | 0.999999 / 0.0117 | 1.000 / 0.0039 |
| state1 cos_min | 0.9979 | 0.9981 |

Two-chunk state-carry (C=256×2 with carried state vs one C=512 pass): the
carry **mechanism works** (chunked ≈ 2-chunk reference, cos_min 0.9886, and
the chunked output also matches the single-512-pass at 0.9886) — the error is
not a state-handling bug.

Isolation ladder (all vs the same MLX reference, C=512):

1. torch fp32 port → MLX: cos_min **0.999999** — the port itself is exact.
2. the *same* Core ML package executed on **CPU** → cos_min **0.99966**.
3. the same package on the **ANE** → cos_min **0.9884**.
4. retry with fp32 SiLU/gating islands → **bit-identical degradation** (0.9884).

The error is uniform across positions (no boundary effect) and is dominated by
ANE fp16 accumulation in the wide reductions (the K=10752 `w2` conv alone
shows 4.9e-3 maxabs on ANE vs 2.9e-4 on CPU). fp16-precision activation
rounding is not the cause; the ANE's fp16 accumulate path is.

## Timing — **FAIL** (gate: C512 p50 ≤ ~24 ms ⇒ ≥ ~7 TFLOPS)

p50 of 50 `predict()` calls after 5 warmups (wall-clock, `CPU_AND_NE`):

| chunk | p50 | TFLOPS |
|---|---|---|
| 128 | 4.45 ms | 1.93 |
| 512 | 35.9 ms | 0.96 |

FLOPs basis: `2 × 33,554,432 matmul params × C` (in_proj 3·2048²,
out_proj 2048², w1+w3 3·2048², w2 2048·10752≈2048²·(10752/2048) — exact
params counted in `e1_gate.py`). The M3 Max GPU sustains 8.3–9.2 TFLOPS on
this model, so the ANE runs these layers **~8× slower per layer** than the
GPU. The 2-layer ANE slice alone (35.9 ms) costs ~60% of the GPU's time for
*all 30 layers* on 512 tokens (~55–62 ms).

## Concurrency — PASS but moot

One process pair, ~10 s per leg: (i) MLX GPU loop over layers 2–29 on a
512-token chunk (lfm25 venv subprocess), (ii) ANE layers 0–1 loop, (iii) both
simultaneously.

- GPU alone: 274.2 ms/chunk · both: 274.4 ms → **0.1% slowdown**
- ANE alone: 35.89 ms/chunk · both: 35.95 ms → **0.2% slowdown**

The engines are fully independent (far under the 10–15% gate), but both
individual gates fail, so independence cannot rescue the idea.

## Verdict: NO-GO

Three independent reasons:

1. **Numerics:** ANE fp16 accumulation loses cos_min 0.988–0.991 against the
   0.9995 gate; the fp32-SiLU retry recovered nothing and CPU execution of the
   identical package is clean, so this is an ANE hardware/executor property,
   not an export bug we can patch with islands.
2. **Throughput:** 0.96 TFLOPS at C=512 vs the ~7 TFLOPS gate. The GPU already
   saturates this model's prefill at 8.2–9.2 TFLOPS (measured, compute-bound),
   so moving layers 0–1 to the ANE removes compute from the faster engine and
   adds it to a ~8× slower one. Even with zero mutual interference this
   *lengthens* 512-token prefill.
3. **Slice size:** the 2-layer slice is not small relative to a full GPU
   prefill, so there is no "tiny tail to hide" — the overlap arithmetic is
   negative before considering handoff costs.

**Do not proceed to E2** (layers 0–5 with attention KV inputs) on this
pattern. The layafast split pays off when the ANE slice is at worst on par
with the GPU per token (Laya's ≤128-token case) or when the GPU is batch-bound
and the batch can be split. LFM2.5's prefill is neither: its conv layers are
wide-GEMM-dominated (2048×10752 SwiGLU), which is exactly the shape the ANE's
fp16 path handles worst. If ANE prefill is still wanted, the only plausible
angle left is weight palettization/fp8-style ANE compression of a *decoder*
slice (laya's M8 direction) plus accepting lower precision — a numerics
conversation, not an engineering one — or profiling whether ANE fares better
on the attention layers (K=2048 reductions only), which E2 would have tested;
the timing gate fails that path first anyway.

## Artifacts

- **Release scope:** This repository contains this summarized report only. The probe scripts, raw `e1_results.json`, generated Core ML packages, and reference-output arrays were local experiment artifacts and are not included. The raw JSON also contains machine-local filesystem paths; do not publish it unchanged.

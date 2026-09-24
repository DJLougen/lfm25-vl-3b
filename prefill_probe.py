"""Is LFM2.5-VL-3B prefill compute-bound on this GPU? Measure GEMM peak and
achieved prefill TFLOPS (text-only LM, chunked prefill like mlx-vlm).

  .venv/bin/python prefill_probe.py --target LFM2.5-VL-3B-MLX-bf16
"""
import argparse, json, time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm import load

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="LFM2.5-VL-3B-MLX-bf16")
ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384, 32768])
ap.add_argument("--steps", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
ap.add_argument("--skip-gemm", action="store_true")
a = ap.parse_args()


def timeit(fn, warm=2, n=5):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return min(ts)


if not a.skip_gemm:
    print("== GEMM x[T,K] @ W[N,K].T (nn.Linear layout)")
    for dt in (mx.bfloat16, mx.float16):
        for T, K, N in ((2048, 2048, 10752), (2048, 10752, 2048), (2048, 2048, 6144), (4096, 2048, 10752)):
            x = mx.random.normal((T, K)).astype(dt); w = mx.random.normal((N, K)).astype(dt)
            s = timeit(lambda: x @ w.T)
            print(f"  {str(dt):14} T={T} K={K} N={N}: {2*T*K*N/s/1e12:6.2f} TFLOPS")

model, _ = load(a.target)
lm = model.language_model
dtype = lm.model.embed_tokens.weight.dtype
layer_params = sum(v.size for k, v in tree_flatten(lm.model.parameters()) if k.startswith("layers.") and k.endswith("weight") and v.ndim == 2)
n_attn = sum(1 for l in lm.model.layers if l.is_attention_layer)
hidden = lm.args.hidden_size
print(f"\ntarget {a.target} dtype {dtype}: layer matmul params {layer_params/1e9:.3f}B, attention layers {n_attn}")


def prefill_flops(L):
    # 2*params per token for projections/MLP + causal attention QK^T and AV (4*hidden per key, avg L/2 keys)
    return 2 * layer_params * L + n_attn * 4 * hidden * (L * L / 2)


ids_all = mx.random.randint(0, 120000, (1, max(a.lengths)))
res = []
for L in a.lengths:
    for step in a.steps:
        if step > L:
            continue
        def run():
            cache = lm.make_cache()
            for i in range(0, L, step):
                lm.model(ids_all[:, i : i + step], cache=cache)  # no lm_head, like prefill
                mx.eval([c.state for c in cache])
            return cache
        run()  # warm
        mx.clear_cache()
        ts = []
        for _ in range(2):
            t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
        s = min(ts)
        r = dict(L=L, step=step, s=round(s, 3), tok_s=round(L / s), tflops=round(prefill_flops(L) / s / 1e12, 2),
                 attn_share=round(n_attn * 4 * hidden * L * L / 2 / prefill_flops(L), 3))
        res.append(r); print(r, flush=True)
json.dump(res, open(f"results/prefill_probe_{a.target.split('-')[-1]}.json", "w"), indent=1)

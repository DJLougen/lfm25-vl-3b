"""Warm, in-process sweep: LFM2.5-VL-3B MLX targets x DSpark drafter variants.

Usage: .venv/bin/python bench.py --targets bf16 8bit --out results/sweep.json
Each config is warmed once, then run for each prompt with `--reps` repetitions.
The run with the lowest end-to-end wall time is retained; its decode throughput,
acceptance, and exact-match result are reported.
"""
import argparse, json, time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

HERE = Path(__file__).parent
PROMPTS = [
    "Describe this image in detail.",
    "Write a long, vivid short story inspired by this image.",
    "List every object visible in this image with its color and position, as a markdown table.",
]
DRAFTERS = {
    "orig": "LFM2.5-VL-3B-DSpark",
    "mlx-bf16": "LFM2.5-VL-3B-DSpark-MLX-bf16",
    "mlx-8bit": "LFM2.5-VL-3B-DSpark-MLX-8bit",
    "mlx-4bit": "LFM2.5-VL-3B-DSpark-MLX-4bit",
}
# (drafter, block, policy); None drafter = baseline
DEFAULT_CONFIGS = [(None, None, None), ("orig", 8, "fixed")] + [
    (d, b, p)
    for d in ("mlx-bf16", "mlx-8bit", "mlx-4bit")
    for b, p in ((4, "fixed"), (8, "fixed"), (8, "adaptive"))
]

ap = argparse.ArgumentParser()
ap.add_argument("--targets", nargs="+", default=["bf16", "8bit"])
ap.add_argument("--image", default=str(HERE / "test.jpg"))
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--out", default=str(HERE / "results" / "sweep.json"))
a = ap.parse_args()

drafters = {k: load_drafter(str(HERE / v)) for k, v in DRAFTERS.items()}


def set_policy(dm, policy):
    if policy == "adaptive":
        dm.prefer_requested_block_size = False
        dm.dflash_initial_block_size = 4
    else:
        dm.prefer_requested_block_size = True
        dm.dflash_initial_block_size = None


results = []
for tname in a.targets:
    model, processor = load(str(HERE / f"LFM2.5-VL-3B-MLX-{tname}"))
    for dm, kind in drafters.values():
        validate_drafter_compatibility(model, dm, kind)

    def run(text, cfg):
        d, block, policy = cfg
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = apply_chat_template(processor, model.config, msgs, add_generation_prompt=True, num_images=1)
        kw = dict(max_tokens=a.max_tokens, temperature=0.0, verbose=False)
        acc = None
        if d is not None:
            dm, kind = drafters[d]
            set_policy(dm, policy)
            kw.update(draft_model=dm, draft_kind=kind, draft_block_size=block)
        t0 = time.perf_counter()
        r = generate(model, processor, prompt, [a.image], **kw)
        wall = time.perf_counter() - t0
        if d is not None:
            al = list(getattr(dm, "accept_lens", []) or [])
            acc = (sum(al) / len(al) + 1) if al else None  # +1 bonus token per round
        return r, wall, acc

    for cfg in DEFAULT_CONFIGS:
        run(PROMPTS[0], cfg)  # warm this shape
    base_text = {}
    for p in PROMPTS:
        for cfg in DEFAULT_CONFIGS:
            best = None
            for _ in range(a.reps):
                r, wall, acc = run(p, cfg)
                if best is None or wall < best["e2e_s"]:
                    best = dict(target=tname, prompt=p[:24], drafter=cfg[0] or "none", block=cfg[1],
                                policy=cfg[2], gen_tokens=r.generation_tokens,
                                decode_tps=round(r.generation_tps, 1), prefill_tps=round(r.prompt_tps, 1),
                                e2e_s=round(wall, 3), accepted_per_round=acc and round(acc, 2),
                                peak_gb=round(r.peak_memory, 2), text=r.text)
            if cfg[0] is None:
                base_text[p] = best["text"]
            best["identical"] = best["text"] == base_text[p]
            results.append(best)
            print({k: v for k, v in best.items() if k != "text"}, flush=True)
    del model, processor
    mx.clear_cache()

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(results, indent=1))
print("wrote", a.out)

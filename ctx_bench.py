"""Context-length ladder benchmark for LFM2.5-VL-3B MLX targets +/- DSpark drafter.

Each prompt = 1 image + an optional real-text document (Pride and Prejudice,
Project Gutenberg) truncated to N target tokens + a task. Appends one JSON line
per measured run to --out.

  .venv/bin/python ctx_bench.py --target LFM2.5-VL-3B-MLX-bf16 \
      --drafter LFM2.5-VL-3B-DSpark-MLX-8bit \
      --configs base 8:fixed:0 8:fixed:2048 4:adaptive:2048 --ctx 0 2048 16384

Config spec: "base" = target only; "<block>:<fixed|adaptive>:<window>" = DSpark
with that verify width, block policy, and draft window (0 = full context).
Within each (ctx, task) the configs run in an order that alternates each rep
(counterbalanced), after a per-config warmup at a short shape.

Metrics: ttft_s = wall to first streamed token (vision + prefill + first draft);
decode_tps = mlx-vlm generation_tps; e2e_s = wall of the whole request;
accepted = mean tokens committed per verify round incl. the bonus token;
peak_gb = mx peak memory for that request only.
"""
import argparse, hashlib, json, time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import apply_chat_template, load, stream_generate
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

from dspark_runtime import configure

HERE = Path(__file__).parent
TASKS = {
    "describe": "Describe the image in detail.",
    "summarize": "Summarize the document above in detail.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--drafter", default=None)
ap.add_argument("--configs", nargs="+", default=["base", "8:fixed:0"])
ap.add_argument("--ctx", type=int, nargs="+", default=[0, 2048, 8192, 16384, 32768, 65536])
ap.add_argument("--tasks", nargs="+", default=list(TASKS))
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--prefill-step", type=int, default=2048)
ap.add_argument("--kv-bits", type=float, default=None,
                help="Quantize target KV cache for long-context tests.")
ap.add_argument("--quantized-kv-start", type=int, default=0)
ap.add_argument("--image", default=str(HERE / "data" / "images" / "coco_cats.jpg"))
ap.add_argument("--book", default=str(HERE / "data" / "book.txt"))
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=str(HERE / "results" / "ctx.jsonl"))
a = ap.parse_args()


def parse(spec):
    if spec == "base":
        return None
    b, p, w = spec.split(":")
    return int(b), p, int(w)


configs = {s: parse(s) for s in a.configs}
model, processor = load(str(HERE / a.target))
tok = processor.tokenizer
dm = kind = None
if a.drafter and any(c for c in configs.values()):
    dm, kind = load_drafter(str(HERE / a.drafter))
    validate_drafter_compatibility(model, dm, kind)

book = Path(a.book).read_text(encoding="utf-8")
start = book.find("Chapter I.")
book_ids = tok.encode(book[start if start > 0 else 0 :], add_special_tokens=False)


def build_prompt(ctx_tokens, task):
    text = TASKS[task]
    if ctx_tokens > 0:
        doc = tok.decode(book_ids[:ctx_tokens])
        text = f"<document>\n{doc}\n</document>\n\n{text}"
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return apply_chat_template(processor, model.config, msgs, add_generation_prompt=True, num_images=1)


def run(prompt, cfg):
    kw = dict(max_tokens=a.max_tokens, temperature=0.0, prefill_step_size=a.prefill_step)
    if a.kv_bits is not None:
        kw.update(kv_bits=a.kv_bits, quantized_kv_start=a.quantized_kv_start)
    if cfg is not None:
        block, policy, window = cfg
        configure(dm, window=window or None, policy=policy)
        kw.update(draft_model=dm, draft_kind=kind, draft_block_size=block)
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    ttft, text, last = None, [], None
    for r in stream_generate(model, processor, prompt, [a.image], **kw):
        if ttft is None:
            ttft = time.perf_counter() - t0
        text.append(r.text)
        last = r
    e2e = time.perf_counter() - t0
    acc = None
    if cfg is not None:
        al = list(getattr(dm, "accept_lens", []) or [])
        acc = round(sum(al) / len(al) + 1, 3) if al else None
    out = "".join(text)
    return dict(
        prompt_tokens=last.prompt_tokens, gen_tokens=last.generation_tokens,
        ttft_s=round(ttft, 3), decode_tps=round(last.generation_tps, 2),
        prefill_tps=round(last.prompt_tps, 1), e2e_s=round(e2e, 3), accepted=acc,
        peak_gb=round(mx.get_peak_memory() / 1e9, 3), kv_bits=a.kv_bits,
        text_sha=hashlib.sha256(out.encode()).hexdigest()[:16], text_head=out[:80],
    )


for cfg in configs.values():  # compile Metal kernels for every config at a short shape
    run(build_prompt(0, "describe"), cfg)

outp = Path(a.out)
outp.parent.mkdir(parents=True, exist_ok=True)
names = list(configs)
for ctx in a.ctx:
    for task in a.tasks:
        if task == "summarize" and ctx == 0:
            continue
        prompt = build_prompt(ctx, task)
        for rep in range(a.reps):
            order = names if rep % 2 == 0 else names[::-1]
            for name in order:
                res = dict(tag=a.tag, target=a.target, drafter=a.drafter if configs[name] else None,
                           config=name, ctx=ctx, task=task, image=Path(a.image).name,
                           prefill_step=a.prefill_step, rep=rep, **run(prompt, configs[name]))
                print(json.dumps(res), flush=True)
                with outp.open("a") as f:
                    f.write(json.dumps(res) + "\n")

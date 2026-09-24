"""Paired target-precision fidelity: bf16 (reference) vs fp16 / 8-bit LFM2.5-VL-3B MLX.

Lossy campaign, separate from the lossless DSpark runtime campaign.
Protocol (quantization-evidence toolkit, qelab):
  1. Reference trajectory = bf16 greedy output (reference-GENERATED text, not truth).
  2. Every model is teacher-forced on the identical token history; complete-vocabulary
     logits at each response position are upcast to fp32 (no clipping).
  3. qelab.metrics.distribution_metrics per position; qelab.statistics.paired_cluster_bootstrap
     with one cluster per workload item (equal cluster weight).
  4. Verified task outcomes where ground truth exists: doc_page transcription CER against the
     source text; chart_bars values found. These come from each model's OWN greedy output.
Exploratory: fewer than 30 independent clusters.

  .venv/bin/python fidelity.py --candidates LFM2.5-VL-3B-MLX-fp16 LFM2.5-VL-3B-MLX-8bit
"""
import argparse, json, re, textwrap
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import apply_chat_template, load, stream_generate
from mlx_vlm.utils import prepare_inputs
from qelab.metrics import distribution_metrics, edit_distance, summarize
from qelab.statistics import paired_cluster_bootstrap

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--reference", default="LFM2.5-VL-3B-MLX-bf16")
ap.add_argument("--candidates", nargs="+", default=["LFM2.5-VL-3B-MLX-fp16", "LFM2.5-VL-3B-MLX-8bit"])
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--long-ctx", type=int, default=4096)
ap.add_argument("--out", default=str(HERE / "results" / "fidelity.json"))
a = ap.parse_args()

work = json.loads((HERE / "data" / "workload.json").read_text())
book = (HERE / "data" / "book.txt").read_text(encoding="utf-8")
start = book.find("Chapter I.")
doc_truth = " ".join(book[start : start + 1400].split())
doc_truth = " ".join(textwrap.wrap(doc_truth, 70))  # the rendered lines, joined

items = [dict(id=w["id"], image=str(HERE / w["image"]), text=w["prompt"]) for w in work]
# Four non-overlapping excerpts are repeated prompts from one source book and
# therefore share a single bootstrap cluster (not four independent samples).
for k, off in enumerate((0, 30000, 60000, 90000)):
    items.append(dict(id=f"longdoc_{k}", image=str(HERE / "data/images/coco_street.jpg"), text=None, book_off=off))
book_token_ids = None


def build(model, processor, it):
    global book_token_ids
    text = it["text"]
    if text is None:
        tok = processor.tokenizer
        if book_token_ids is None:
            book_token_ids = tok.encode(book[start:], add_special_tokens=False)
        ids = book_token_ids[it["book_off"] : it["book_off"] + a.long_ctx]
        text = f"<document>\n{tok.decode(ids)}\n</document>\n\nSummarize the document above in detail."
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return apply_chat_template(processor, model.config, msgs, add_generation_prompt=True, num_images=1)


def greedy(model, processor, it):
    prompt = build(model, processor, it)
    toks, text = [], []
    for r in stream_generate(model, processor, prompt, [it["image"]], max_tokens=a.max_tokens, temperature=0.0):
        toks.append(int(r.token)); text.append(r.text)
    return prompt, toks, "".join(text)


def forced_logits(model, processor, prompt, image, gen):
    inp = prepare_inputs(processor, images=[image], prompts=prompt,
                         image_token_index=model.config.image_token_index, add_special_tokens=False)
    ids = inp["input_ids"]
    full = mx.concatenate([ids, mx.array([gen], dtype=ids.dtype)], axis=1)
    emb = model.get_input_embeddings(full, inp["pixel_values"], spatial_shapes=inp.get("spatial_shapes"),
                                     pixel_attention_mask=inp.get("pixel_attention_mask")).inputs_embeds
    lm = model.language_model
    cache = lm.make_cache()
    P, G = ids.shape[1], len(gen)
    outs = []
    for i in range(0, full.shape[1], 2048):
        h = lm.model(full[:, i : i + 2048], cache=cache, input_embeddings=emb[:, i : i + 2048])
        lo, hi = max(i, P - 1), min(i + h.shape[1], P + G - 1)  # positions predicting gen tokens
        if hi > lo:
            outs.append(lm.model.embed_tokens.as_linear(h[:, lo - i : hi - i]).astype(mx.float32))
        mx.eval(outs, [c.state for c in cache])
    return np.array(mx.concatenate(outs, axis=1)[0])


def task_scores(it_id, text):
    if it_id == "doc_page":
        hyp = " ".join(text.split())
        return {"doc_cer": edit_distance(doc_truth, hyp) / len(doc_truth)}
    if it_id == "chart_bars":
        found = [v for v in ("42", "57", "35", "71", "64") if re.search(rf"\b{v}\b", text)]
        return {"chart_values_found": len(found)}
    return {}


# 1. reference trajectories + reference logits
model, processor = load(str(HERE / a.reference))
ref = {}
for it in items:
    prompt, toks, text = greedy(model, processor, it)
    lg = forced_logits(model, processor, prompt, it["image"], toks)
    lg2 = forced_logits(model, processor, prompt, it["image"], toks)  # base/base repeatability
    if len(toks) == 0 or lg.shape[0] != len(toks):
        raise SystemExit(f"reference teacher-forcing shape mismatch on {it['id']}")
    if not np.array_equal(lg, lg2):
        raise SystemExit(f"reference logits are not repeatable on {it['id']}")
    ref[it["id"]] = dict(prompt=prompt, toks=toks, text=text, logits=lg,
                         repeat_bitwise_equal=bool(np.array_equal(lg, lg2)),
                         forced_argmax_matches_greedy=float(np.mean(lg.argmax(1) == np.array(toks))),
                         tasks=task_scores(it["id"], text))
    print(it["id"], len(toks), "repeat-equal", ref[it["id"]]["repeat_bitwise_equal"],
          "argmax==greedy", ref[it["id"]]["forced_argmax_matches_greedy"], ref[it["id"]]["tasks"], flush=True)
del model; mx.clear_cache()

cluster_for = lambda item_id: "public_domain_book" if item_id.startswith("longdoc_") else item_id
report = {"reference": a.reference, "clusters": {it["id"]: cluster_for(it["id"]) for it in items},
          "n_clusters": len(set(cluster_for(it["id"]) for it in items)),
          "status": "exploratory (<30 independent clusters)", "reference_tasks": {k: v["tasks"] for k, v in ref.items()},
          "reference_repeatability": {k: v["repeat_bitwise_equal"] for k, v in ref.items()},
          "reference_forced_argmax_match": {k: v["forced_argmax_matches_greedy"] for k, v in ref.items()},
          "candidates": {}}
KEYS = ["kl_reference_to_candidate", "js_divergence", "total_variation", "top1_agreement",
        "topk_overlap_fraction", "delta_nll", "candidate_rank_of_reference_top1"]
for cand in a.candidates:
    model, processor = load(str(HERE / cand))
    per = {k: [] for k in KEYS}; groups = []; ref_nll = []; cand_nll = []; own = {}
    for it in items:
        r = ref[it["id"]]
        cl = forced_logits(model, processor, r["prompt"], it["image"], r["toks"])
        if not np.isfinite(cl).all():
            raise SystemExit(f"non-finite logits from {cand} on {it['id']} (recorded failure)")
        m = distribution_metrics(r["logits"], cl, observed_tokens=np.array(r["toks"]))
        for k in KEYS:
            per[k].extend(m[k].tolist())
        ref_nll.extend(m["reference_nll"].tolist()); cand_nll.extend(m["candidate_nll"].tolist())
        groups.extend([cluster_for(it["id"])] * len(r["toks"]))
        _, toks, text = greedy(model, processor, it)
        n = min(len(toks), len(r["toks"]))
        mism = next((i for i in range(n) if toks[i] != r["toks"][i]), None)
        own[it["id"]] = dict(identical_to_reference=toks == r["toks"], first_mismatch=mism, tasks=task_scores(it["id"], text))
        print(cand, it["id"], {k: round(float(np.mean(m[k])), 5) for k in KEYS[:4]}, own[it["id"]], flush=True)
    summ = {k: summarize(np.array(v)) for k, v in per.items()}
    boot = {
        "delta_nll": paired_cluster_bootstrap(ref_nll, cand_nll, groups),
        "top1_agreement_minus_1": paired_cluster_bootstrap(np.ones(len(groups)), per["top1_agreement"], groups),
    }
    report["candidates"][cand] = {"per_position_summary": summ, "cluster_bootstrap": boot, "own_greedy": own,
                                  "n_positions": len(groups)}
    Path(a.out).write_text(json.dumps(report, indent=1, default=float))
    print("checkpointed", a.out, flush=True)
    del model; mx.clear_cache()

Path(a.out).write_text(json.dumps(report, indent=1, default=float))
print("wrote", a.out)

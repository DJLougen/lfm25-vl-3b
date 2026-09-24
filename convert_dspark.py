"""Convert LiquidAI/LFM2.5-VL-3B-DSpark (HF/SGLang bf16) into an MLX-native drafter folder.

The drafter shares embed_tokens / lm_head with its target at bind() time, so the
output holds only the 4 draft layers + fc/hidden_norm + Markov/confidence heads.

  .venv/bin/python convert_dspark.py --out LFM2.5-VL-3B-DSpark-MLX-bf16
  .venv/bin/python convert_dspark.py --out LFM2.5-VL-3B-DSpark-MLX-8bit --q-bits 8
  .venv/bin/python convert_dspark.py --out LFM2.5-VL-3B-DSpark-MLX-4bit --q-bits 4

Quantization touches only the draft transformer linears (layers.*, fc); the
Markov head, confidence head, and norms stay bf16.
"""
import argparse, json, shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.speculative.drafters import load_drafter

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="LFM2.5-VL-3B-DSpark")
ap.add_argument("--out", required=True)
ap.add_argument("--q-bits", type=int, default=None)
ap.add_argument("--q-group-size", type=int, default=64)
ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
model, kind = load_drafter(str(src))
assert type(model).__name__ == "DSparkDraftModel" and kind == "dflash", (type(model), kind)
model.set_dtype(getattr(mx, a.dtype))

config = json.loads((src / "config.json").read_text())
config["dtype"] = a.dtype
if a.q_bits:
    def pred(path, m):
        return (
            isinstance(m, nn.Linear)
            and (path.startswith("layers.") or path == "fc")
            and m.weight.shape[-1] % a.q_group_size == 0
        )
    nn.quantize(model, group_size=a.q_group_size, bits=a.q_bits, class_predicate=pred)
    q = {"group_size": a.q_group_size, "bits": a.q_bits, "mode": "affine"}
    config["quantization"] = q
    config["quantization_config"] = q

weights = dict(tree_flatten(model.parameters()))
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
(out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
for f in ("LICENSE",):
    shutil.copy(src / f, out / f)
dt = {"bfloat16": "bf16", "float16": "fp16"}[a.dtype]
bits = f"{a.q_bits}-bit affine g{a.q_group_size} (draft layers + fc), {dt} rest" if a.q_bits else dt
(out / "README.md").write_text(
    f"""---
library_name: mlx
base_model: LiquidAI/LFM2.5-VL-3B-DSpark
license: other
license_name: lfm1.0
license_link: LICENSE
tags: [mlx, mlx_vlm, dspark, speculative-decoding, draft-model, lfm2-vl]
---
# LFM2.5-VL-3B-DSpark MLX ({bits})

MLX-native conversion of [LiquidAI/LFM2.5-VL-3B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark)
(rev af77e9306a26e8625fde74d2a3051ab6d21bd955). Speculative drafter for LFM2.5-VL-3B MLX targets.
Markov/confidence heads and norms kept unquantized ({dt}). Needs mlx-vlm >= 0.7.2.

```bash
mlx_vlm.generate --model LiquidAI/LFM2.5-VL-3B-MLX-bf16 --draft-model <this folder> \\
  --draft-block-size 8 --temperature 0 --image img.jpg --prompt "Describe this image."
```
"""
)
n = sum(v.nbytes for v in weights.values())
print(f"wrote {out}: {len(weights)} tensors, {n/1e6:.1f} MB, {bits}")

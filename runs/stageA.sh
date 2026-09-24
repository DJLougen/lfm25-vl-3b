#!/bin/bash
# Stage A: target/drafter dtype pairing at ctx 0 and 16k, block 8 vs target-only.
cd "$(dirname "$0")/.."
run(){ .venv/bin/python ctx_bench.py --target "$1" --drafter "$2" --blocks 0 8 --ctx 0 16384 --reps 2 --tag stageA --out results/stageA.jsonl 2>&1 | grep '^{' | wc -l; }
run LFM2.5-VL-3B-MLX-bf16 LFM2.5-VL-3B-DSpark-MLX-bf16
run LFM2.5-VL-3B-MLX-fp16 LFM2.5-VL-3B-DSpark-MLX-fp16
run LFM2.5-VL-3B-MLX-fp16 LFM2.5-VL-3B-DSpark-MLX-8bit-fp16
run LFM2.5-VL-3B-MLX-fp16 LFM2.5-VL-3B-DSpark-MLX-8bit
echo ALLDONE

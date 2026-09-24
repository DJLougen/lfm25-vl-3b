# Agent entrypoint

This repository contains benchmark code and provenance for running Liquid AI's LFM2.5-VL-3B target and DSpark speculative drafter on Apple Silicon. The large target and draft weight directories are intentionally excluded from source control; obtain them from the linked Hugging Face repositories or the upstream Liquid AI repositories and verify their hashes against `PROVENANCE.json`.

## Environment and smoke test

Use Python 3.12 on Apple Silicon with the pinned dependencies in `requirements.txt`. Install them into a project virtual environment, download the required model folders, and run the image-generation command in `README.md`. DSpark decoding is greedy-only in the tested mlx-vlm version, so use `--temperature 0`.

## Benchmarks

- `bench.py`: short-prompt precision/draft screen.
- `ctx_bench.py`: paired context sweep; raw measurements are under `results/`.
- `fidelity.py`: exploratory target-precision fidelity analysis. It requires the separate `quantization-evidence` qelab package, which is intentionally not part of this public repository.
- `make_workload.py`: reconstructs the local benchmark workload; downloaded COCO images are not committed.
- `ane/E1_REPORT.md`: archived NO-GO report. The experimental Core ML package files and scripts with machine-local paths are intentionally excluded.

Do not expand the benchmark claims beyond the sample and configurations recorded in the raw result files and `PROVENANCE.json`. In particular, the long-context results do not support enabling DSpark speculation at 32k or 64k, and the 8-bit target showed no benefit from the tested quantized KV-cache configuration.

## License boundaries

The root source code is Apache-2.0. The separately published DSpark draft weights are derivatives of Liquid AI model weights and retain the upstream LFM Open License v1.0. Do not apply the source license to model weights.

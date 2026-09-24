Each prompt/configuration had two repetitions; the row retained is the one with the lowest end-to-end wall time. Decode rates below are that selected run's rates. Ratios are averaged across the three prompts, so best-of-two selection may be optimistic.

| target | drafter | block/policy | decode tok/s (3 prompts) | decode speedup (mean) | e2e speedup (mean) | accepted/round |
|---|---|---|---|---|---|---|
| bf16 | none | - | 43.2 / 41.9 / 45.8 | 1.00x | 1.00x | - |
| bf16 | orig | 8 fixed | 140.3 / 109.1 / 137.4 | 2.95x | 2.40x | 4.30 |
| bf16 | mlx-bf16 | 4 fixed | 98.1 / 86.1 / 109.9 | 2.24x | 1.94x | 3.00 |
| bf16 | mlx-bf16 | 8 fixed | 142.6 / 110.3 / 139.1 | 2.99x | 2.41x | 4.30 |
| bf16 | mlx-bf16 | 8 adaptive | 115.0 / 91.6 / 111.1 | 2.42x | 2.08x | 3.27 |
| bf16 | mlx-8bit | 4 fixed | 106.4 / 84.4 / 111.2 | 2.30x | 2.00x | 2.99 |
| bf16 | mlx-8bit | 8 fixed | 141.8 / 108.7 / 145.3 | 3.02x | 2.44x | 4.30 |
| bf16 | mlx-8bit | 8 adaptive | 116.4 / 93.6 / 117.9 | 2.50x | 2.13x | 3.25 |
| bf16 | mlx-4bit | 4 fixed | 111.3 / 87.0 / 104.4 | 2.31x | 2.01x | 2.96 |
| bf16 | mlx-4bit | 8 fixed | 133.6 / 104.6 / 127.0 | 2.79x | 2.31x | 4.09 |
| bf16 | mlx-4bit | 8 adaptive | 107.7 / 102.0 / 95.3 | 2.34x | 2.04x | 3.07 |
| 8bit | none | - | 69.0 / 78.1 / 77.7 | 1.00x | 1.00x | - |
| 8bit | orig | 8 fixed | 76.9 / 52.7 / 65.2 | 0.88x | 0.90x | 4.07 |
| 8bit | mlx-bf16 | 4 fixed | 88.5 / 72.6 / 76.6 | 1.07x | 1.06x | 2.97 |
| 8bit | mlx-bf16 | 8 fixed | 77.4 / 52.9 / 65.5 | 0.88x | 0.90x | 4.07 |
| 8bit | mlx-bf16 | 8 adaptive | 85.7 / 74.2 / 74.9 | 1.05x | 1.05x | 3.22 |
| 8bit | mlx-8bit | 4 fixed | 93.5 / 74.2 / 78.9 | 1.11x | 1.09x | 2.99 |
| 8bit | mlx-8bit | 8 fixed | 77.2 / 53.7 / 66.6 | 0.89x | 0.91x | 4.07 |
| 8bit | mlx-8bit | 8 adaptive | 83.5 / 72.5 / 76.4 | 1.04x | 1.04x | 3.34 |
| 8bit | mlx-4bit | 4 fixed | 93.0 / 70.2 / 76.1 | 1.08x | 1.06x | 2.92 |
| 8bit | mlx-4bit | 8 fixed | 71.2 / 49.9 / 66.7 | 0.84x | 0.87x | 3.93 |
| 8bit | mlx-4bit | 8 adaptive | 79.1 / 70.7 / 71.8 | 0.99x | 0.99x | 3.01 |

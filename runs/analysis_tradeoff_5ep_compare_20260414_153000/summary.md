# Trade-off Analysis

Reference cost setting: D1 test samples, batch size 8.
Reference compute setting: D1 encoder MACs per sample.
Reference performance setting: mean NMSE over D1, D5, D9, D15.

## Training Speed

- total_time_hours: 5.276
- minutes_per_epoch: 63.31
- train_samples_per_second_est: 37.91
- all_samples_per_second_est: 39.59
- model_label: Stru-JEPA-Base
- subnet_sampling_mode: anchor_random
- raw_sizes_plotted: tiny, little, small, base
- widths: 1, 0.5, 0.125
- depths: 1, 0.5, 0.166667

## Stru-JEPA Temporal Prediction (D1-D16)

| Dataset | Stru-JEPA NMSE |
| --- | ---: |
| D1 | 0.074 |
| D2 | 0.247 |
| D3 | 0.016 |
| D4 | 0.040 |
| D5 | 0.455 |
| D6 | 0.093 |
| D7 | 0.073 |
| D8 | 0.018 |
| D9 | 0.315 |
| D10 | 0.500 |
| D11 | 0.209 |
| D12 | 0.023 |
| D13 | 0.523 |
| D14 | 0.342 |
| D15 | 0.027 |
| D16 | 0.298 |
| Average | 0.203 |

## Stru-JEPA Frequency Prediction (D1-D16)

| Dataset | Stru-JEPA NMSE |
| --- | ---: |
| D1 | 0.306 |
| D2 | 0.174 |
| D3 | 0.023 |
| D4 | 0.065 |
| D5 | 0.138 |
| D6 | 0.089 |
| D7 | 0.083 |
| D8 | 0.052 |
| D9 | 0.411 |
| D10 | 0.094 |
| D11 | 0.210 |
| D12 | 0.023 |
| D13 | 0.077 |
| D14 | 0.376 |
| D15 | 0.021 |
| D16 | 0.243 |
| Average | 0.149 |

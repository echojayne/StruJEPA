# 5-Epoch Vs New Runs

Lower NMSE is better. `20ep` values are only available for completed new runs (`small`, `little`). `base` and `tiny` new long runs were stopped before checkpoint export, so there is no final 20-epoch test point for them.

## Full Subnet `w1_d1`

| Size | Raw Temporal | 5ep Temporal | 20ep Temporal | Raw Fre | 5ep Fre | 20ep Fre |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 0.316 | 0.358 | - | 0.343 | 0.367 | - |
| little | 0.270 | 0.274 | 0.287 | 0.260 | 0.282 | 0.279 |
| small | 0.245 | 0.238 | 0.251 | 0.239 | 0.224 | 0.233 |
| base | 0.237 | 0.218 | - | 0.232 | 0.219 | - |

## Curve-Wise 5ep Vs 20ep (`small`, `little`)

| Size | Task | Subnets | 5ep Better | 20ep Better | Mean(20ep-5ep) | Worst 20ep Regression | Best 20ep Gain |
|---|---|---:|---:|---:|---:|---:|---:|
| small | temporal | 9 | 4 | 5 | 0.001 | 0.013 | -0.004 |
| small | fre | 9 | 6 | 3 | 0.002 | 0.009 | -0.003 |
| little | temporal | 9 | 1 | 8 | -0.021 | 0.013 | -0.037 |
| little | fre | 9 | 0 | 9 | -0.010 | -0.002 | -0.016 |

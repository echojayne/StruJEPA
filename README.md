# StruJEPA

This repository maintains one current StruJEPA method:
`matrix_transformer_full_lpeadd`.

The method trains a shared matrix-Transformer completion module for missing
width operators, using Gaussian-CDF depth encoding, and then trains deployable
width/depth subnets with task loss. The released current configurations use:

- `completion.mode: width_operator_completion`
- `completion.predictor_layout: matrix_transformer_full`
- `completion.depth_encoding: gaussian_cdf`
- task-only subnet training, equivalent to `lambda_output=0` and
  `lambda_repr=0`; those legacy knobs are no longer exposed

The completion module is training-only. Exported or evaluated elastic subnets
do not require it at inference time.

## Layout

- `elastic_method/`: reusable elasticization and current StruJEPA trainer
- `integrations/`: BeamFormer, WiFo, channel-estimation, traffic, and DINOv3
  adapters
- `configs/`: one current configuration per maintained model family
- `scripts/`: maintained training entrypoints

Git metadata, runtime outputs, figures, portable deliverables, checkpoints,
dated launchers, ablations, and OAI assets are intentionally outside this core
method directory.

## Installation

```bash
python -m pip install -e ".[dev]"
export AI_RAN_BENCHMARK_ROOT=/path/to/ai_ran_benchmarks
```

Additional task-specific dependencies are available through the `wifo`,
`channel`, `traffic`, and `dinov3` extras.

## Current Training Entrypoints

WiFo:

```bash
python scripts/run_wifo_strujepa.py \
  --config configs/channel_prediction/current_wifo.json
```

AdaFortiTran and A-MMSE:

```bash
python scripts/run_channel_estimation_strujepa.py \
  --config configs/channel_estimation/current_adafortitran.yaml

python scripts/run_channel_estimation_strujepa.py \
  --config configs/channel_estimation/current_ammse.yaml
```

iTransformer:

```bash
python scripts/run_traffic_forecasting_strujepa.py \
  --config configs/traffic_forecasting/current_itransformer.yaml
```

DINOv3/ImageNet:

```bash
export DINOV3_ROOT=/path/to/dinov3
export IMAGENET_ROOT=/path/to/imagenet
export DINOV3_HPLUS_WEIGHTS=/path/to/dinov3_vith16plus.pth
python scripts/run_dinov3_imagenet_strujepa.py \
  --config configs/dinov3_imagenet/current_dinov3.json
```

The DINOv3 integration remains experimental. The former `matrix_row_12pt`
recipe is not part of the maintained method.

BeamFormer:

```bash
python scripts/run_beamformer_strujepa.py \
  --config configs/beamformer/current_beamformer.json
```

## Validation

```bash
python -m unittest discover -s elastic_method/tests -p "test_*.py"
python -m compileall -q elastic_method integrations scripts
```

See [docs/MIGRATION.md](docs/MIGRATION.md) for removed interfaces and paths.

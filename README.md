# StruJEPA

StruJEPA is a structure-aware elastic training method for transformer backbones. This repository contains:

- `elastic_method/`
  - the reusable elasticization and alignment framework
- `WIFO/`
  - the WiFo integration, training entrypoints, evaluation scripts, and reference materials
- `runs/`
  - retained checkpoints, logs, and paper-style trade-off analysis outputs

## Repository Layout

- `elastic_method/`
  - core elastic model wrapper, adapters, subnet logic, and tests
- `WIFO/src/`
  - WiFo model code plus StruJEPA training and evaluation scripts
  - main entry: `WIFO/src/strujepa_main.py`
- `runs/analysis_tradeoff_5ep_compare_20260414_153000/`
  - retained figures, CSVs, and markdown summaries for the current WiFo trade-off study
- `runs/wifo_*_anchor_random_*/`
  - retained random-subnet checkpoints and training histories

## Data And Weights

The local working copy contains WiFo datasets and pretrained weights, but they are intentionally excluded from version control because they are large local assets:

- `WIFO/dataset/`
- `WIFO/dataset4train/`
- `WIFO/weights/`

Training and evaluation scripts expect those paths to exist locally.

## Typical Commands

Train WiFo with StruJEPA:

```bash
python WIFO/src/strujepa_main.py \
  --dataset D1 \
  --train_data_root /abs/path/to/WIFO/dataset4train \
  --val_data_root /abs/path/to/WIFO/dataset4train \
  --file_load_path /abs/path/to/WIFO/weights/wifo_tiny.pkl
```

Run integration tests:

```bash
python -m unittest elastic_method.tests.test_wifo_vit
python -m unittest elastic_method.tests.test_wifo_strujepa
```

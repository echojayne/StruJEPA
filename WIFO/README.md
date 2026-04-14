# WiFo Integration

This directory contains the WiFo-specific part of the StruJEPA repository.

## Contents

- `src/`
  - original WiFo model code plus StruJEPA training, evaluation, and plotting scripts
  - main entries:
    - `main.py`: original WiFo inference/evaluation entry
    - `strujepa_main.py`: StruJEPA training entry
    - `analyze_tradeoff.py`: paper-style trade-off evaluation
- `reference/`
  - local reference PDFs used during analysis
- `requirements.txt`
  - WiFo-side Python dependencies

## Local Assets

These directories are kept locally but ignored by git:

- `dataset/`
- `dataset4train/`
- `weights/`

The compatibility path `src/weights` still points to `../weights`.

## Typical Commands

Run WiFo StruJEPA training:

```bash
python src/strujepa_main.py \
  --dataset D1 \
  --train_data_root /abs/path/to/WIFO/dataset4train \
  --val_data_root /abs/path/to/WIFO/dataset4train \
  --file_load_path /abs/path/to/WIFO/weights/wifo_tiny.pkl
```

Run original WiFo inference:

```bash
python src/main.py --device_id 0 --size base --mask_strategy_random none --mask_strategy temporal --dataset D17 --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D
```

# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from elastic_wifo import (
    build_headwise_width_multipliers,
    build_layerwise_depth_multipliers,
    elasticize_wifo,
)
from model import WiFo_model
from strujepa_data import build_loader
from strujepa_recipe_trainer import WiFoStruJEPATrainer
from strujepa_wifo import WiFoStruJEPACallback, parse_int_string, parse_multiplier_string
from utils import add_dict_to_argparser

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from elastic_method import MethodConfig


def setup_init(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dev(device_id: str = "0") -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def create_argparser():
    defaults = dict(
        dataset="*".join(f"D{i}" for i in range(1, 17)),
        train_data_root="../dataset4train",
        val_data_root="../dataset4train",
        train_split="train",
        val_split="val",
        save_dir="../experiments/strujepa_wifo_base",
        num_workers=4,
        seed=100,
        epochs=5,
        batch_size=8,
        lr=1e-4,
        weight_decay=0.05,
        size="base",
        patch_size=4,
        t_patch_size=4,
        no_qkv_bias=0,
        pos_emb="SinCos_3D",
        mask_ratio=0.5,
        mask_strategy="random",
        task_specs="random:0.85,temporal:0.5,fre:0.5",
        width_multipliers="1.0,0.5,0.125",
        depth_multipliers="1.0,0.5,0.166667",
        use_headwise_width_ladder=False,
        use_layerwise_depth_ladder=False,
        active_head_values="",
        active_layer_values="",
        min_active_heads=1,
        min_active_layers=1,
        width_only_epochs=0,
        supervised_weight=0.75,
        lambda_output=1.5,
        lambda_repr=0.25,
        enable_output_alignment=True,
        enable_repr_alignment=True,
        use_ema_full_view=True,
        ema_momentum=0.995,
        objective_mode="full_plus_mean_subnets",
        subnet_sampling_mode="anchor_random",
        random_subnets_per_batch=1,
        validate_every=5,
        log_every_batches=500,
        align_masked_only=True,
        file_load_path="",
        device_id="0",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    setup_init(int(args.seed))
    device = dev(args.device_id)
    dataset_names = [name for name in str(args.dataset).split("*") if name]
    if not args.file_load_path:
        default_weight_path = Path(__file__).resolve().parent.parent / "weights" / f"wifo_{args.size}.pkl"
        if default_weight_path.exists():
            args.file_load_path = str(default_weight_path)

    train_loader = build_loader(
        dataset_names,
        root=args.train_data_root,
        split=args.train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        dataset_names,
        root=args.val_data_root,
        split=args.val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    base_model = WiFo_model(args=args)
    if args.file_load_path:
        state_dict = torch.load(args.file_load_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        base_model.load_state_dict(state_dict, strict=False)

    active_head_values = parse_int_string(args.active_head_values)
    active_layer_values = parse_int_string(args.active_layer_values)
    width_multipliers = (
        build_headwise_width_multipliers(
            base_model,
            active_head_values=active_head_values if active_head_values else None,
            min_active_heads=int(args.min_active_heads),
        )
        if args.use_headwise_width_ladder or active_head_values
        else parse_multiplier_string(args.width_multipliers)
    )
    depth_multipliers = (
        build_layerwise_depth_multipliers(
            base_model,
            active_layer_values=active_layer_values if active_layer_values else None,
            min_active_layers=int(args.min_active_layers),
        )
        if args.use_layerwise_depth_ladder or active_layer_values
        else parse_multiplier_string(args.depth_multipliers)
    )
    print(f"width_multipliers={width_multipliers}")
    print(f"depth_multipliers={depth_multipliers}")

    elastic_model = elasticize_wifo(
        base_model,
        width_multipliers=width_multipliers,
        depth_multipliers=depth_multipliers,
        width_only_epochs=args.width_only_epochs,
        copy_model=True,
    )
    callback = WiFoStruJEPACallback(
        representation_dim=elastic_model.model.embed_dim,
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy,
        task_specs=args.task_specs,
        base_seed=args.seed,
        align_masked_only=args.align_masked_only,
    )
    trainer = WiFoStruJEPATrainer(
        elastic_model,
        callback,
        spec=elastic_model.spec,
        config=MethodConfig(
            supervised_weight=args.supervised_weight,
            lambda_output=args.lambda_output,
            lambda_repr=args.lambda_repr,
            enable_output_alignment=args.enable_output_alignment,
            enable_repr_alignment=args.enable_repr_alignment,
            use_ema_full_view=args.use_ema_full_view,
            ema_momentum=args.ema_momentum,
        ),
        random_subnets_per_batch=args.random_subnets_per_batch,
        sampling_seed=args.seed,
        validate_every=args.validate_every,
        log_every_batches=args.log_every_batches,
        subnet_sampling_mode=args.subnet_sampling_mode,
        objective_mode=args.objective_mode,
        device=device,
    )
    optimize_params = list(trainer.model.parameters())
    if trainer.mask_module is not None:
        optimize_params.extend(trainer.mask_module.parameters())
    trainer.optimizer = torch.optim.AdamW(optimize_params, lr=args.lr, weight_decay=args.weight_decay)

    history = trainer.fit(train_loader, epochs=args.epochs, val_loader=val_loader)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "mask_module_state_dict": trainer.mask_module.state_dict() if trainer.mask_module is not None else None,
        "history": history,
        "args": vars(args),
    }
    torch.save(checkpoint, save_dir / "strujepa_wifo_last.pt")
    with (save_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    for record in history:
        epoch = int(record["epoch"])
        train_loss = record.get("train_loss", float("nan"))
        train_nmse = record.get("train_nmse", float("nan"))
        val_loss = record.get("val_loss", float("nan"))
        val_nmse = record.get("val_nmse", float("nan"))
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_nmse={train_nmse:.6f} "
            f"val_loss={val_loss:.6f} val_nmse={val_nmse:.6f}"
        )


if __name__ == "__main__":
    main()

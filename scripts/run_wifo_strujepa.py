# coding=utf-8
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.wifo.benchmark_paths import resolve_wifo_benchmark_paths

WIFO_PATHS = resolve_wifo_benchmark_paths()

from elastic_method import MethodConfig


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def add_dict_to_argparser(parser: argparse.ArgumentParser, defaults: dict[str, object]) -> None:
    for key, value in defaults.items():
        value_type = _parse_bool if isinstance(value, bool) else type(value)
        parser.add_argument(f"--{key.replace('_', '-')}", default=value, type=value_type)


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
        train_data_root=str(WIFO_PATHS.train_val_data),
        val_data_root=str(WIFO_PATHS.train_val_data),
        train_split="train",
        val_split="val",
        save_dir=str(ROOT / "runs" / "strujepa_wifo_base"),
        num_workers=4,
        seed=100,
        epochs=10,
        batch_size=8,
        lr=1e-4,
        lr_gamma=1.0,
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
        supervised_weight=1.0,
        completion_enabled=True,
        completion_mode="width_operator_completion",
        completion_stage_epochs="5,5",
        completion_depth_encoding="gaussian_cdf",
        completion_gaussian_mu=0.37790416,
        completion_gaussian_sigma=0.2029252,
        completion_lambda_weight=1.0,
        completion_lambda_attn_residual=0.25,
        completion_lambda_ffn_residual=0.25,
        completion_predictor_hidden_dim=256,
        completion_predictor_layers=1,
        completion_predictor_layout="matrix_transformer_full",
        objective_mode="full_plus_mean_subnets",
        subnet_sampling_mode="anchor_random",
        random_subnets_per_batch=1,
        validate_every=5,
        checkpoint_every=1,
        log_every_batches=500,
        file_load_path="",
        resume_from_checkpoint=False,
        resume_optimizer_state=True,
        device_id="0",
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/channel_prediction/current_wifo.json",
    )
    known, _ = config_parser.parse_known_args()
    if known.config.is_file():
        payload = json.loads(known.config.read_text(encoding="utf-8"))
        unknown = sorted(set(payload) - set(defaults))
        if unknown:
            raise ValueError(f"unknown WiFo config keys: {unknown}")
        defaults.update(payload)
    parser = argparse.ArgumentParser(parents=[config_parser])
    add_dict_to_argparser(parser, defaults)
    return parser


def parse_completion_stage_epochs(raw: str) -> dict[str, int]:
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if len(values) != 2:
        raise ValueError("completion_stage_epochs must be 'warmup_completion,subnet_training'")
    return {
        "warmup_completion": values[0],
        "subnet_training": values[1],
    }


def normalize_wifo_state_dict_for_base_model(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(key.startswith("model.") for key in state_dict):
        return {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    return state_dict


def main():
    args = create_argparser().parse_args()
    from integrations.wifo.benchmark_paths import ensure_wifo_source_path
    from integrations.wifo.elastic_wifo import (
        build_headwise_width_multipliers,
        build_layerwise_depth_multipliers,
        elasticize_wifo,
    )
    from integrations.wifo.strujepa_data import build_loader
    from integrations.wifo.strujepa_recipe_trainer import WiFoStruJEPATrainer
    from integrations.wifo.strujepa_wifo import WiFoTaskAdapter, parse_int_string, parse_multiplier_string

    ensure_wifo_source_path(WIFO_PATHS)
    from model import WiFo_model

    setup_init(int(args.seed))
    device = dev(args.device_id)
    dataset_names = [name for name in str(args.dataset).split("*") if name]
    if not args.file_load_path:
        default_weight_path = WIFO_PATHS.original_weights / f"wifo_{args.size}.pkl"
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
    loaded_checkpoint = None
    if args.file_load_path:
        loaded_checkpoint = torch.load(args.file_load_path, map_location="cpu")
        state_dict = loaded_checkpoint
        if isinstance(loaded_checkpoint, dict) and "model_state_dict" in loaded_checkpoint:
            state_dict = loaded_checkpoint["model_state_dict"]
        state_dict = normalize_wifo_state_dict_for_base_model(state_dict)
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        print(
            f"loaded file_load_path={args.file_load_path} "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )

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
    task_adapter = WiFoTaskAdapter(
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy,
        task_specs=args.task_specs,
        base_seed=args.seed,
    )
    trainer = WiFoStruJEPATrainer(
        elastic_model,
        task_adapter,
        spec=elastic_model.spec,
        config=MethodConfig(
            supervised_weight=args.supervised_weight,
            completion={
                "enabled": bool(args.completion_enabled),
                "mode": str(args.completion_mode),
                "stage_epochs": parse_completion_stage_epochs(args.completion_stage_epochs),
                "depth_encoding": str(args.completion_depth_encoding),
                "gaussian_mu": float(args.completion_gaussian_mu),
                "gaussian_sigma": float(args.completion_gaussian_sigma),
                "lambda_weight": float(args.completion_lambda_weight),
                "lambda_attn_residual": float(args.completion_lambda_attn_residual),
                "lambda_ffn_residual": float(args.completion_lambda_ffn_residual),
                "predictor_hidden_dim": int(args.completion_predictor_hidden_dim),
                "predictor_layers": int(args.completion_predictor_layers),
                "predictor_layout": str(args.completion_predictor_layout),
            },
        ),
        random_subnets_per_batch=args.random_subnets_per_batch,
        sampling_seed=args.seed,
        validate_every=args.validate_every,
        log_every_batches=args.log_every_batches,
        subnet_sampling_mode=args.subnet_sampling_mode,
        objective_mode=args.objective_mode,
        device=device,
    )
    if isinstance(loaded_checkpoint, dict):
        if loaded_checkpoint.get("model_state_dict") is not None:
            model_missing, model_unexpected = trainer.model.load_state_dict(
                loaded_checkpoint["model_state_dict"],
                strict=False,
            )
            print(
                "loaded elastic model state "
                f"missing_keys={len(model_missing)} unexpected_keys={len(model_unexpected)}"
            )
        if trainer.mask_module is not None and loaded_checkpoint.get("mask_module_state_dict") is not None:
            trainer.mask_module.load_state_dict(loaded_checkpoint["mask_module_state_dict"], strict=False)
        if trainer.completion_module is not None and loaded_checkpoint.get("completion_state_dict") is not None:
            trainer.completion_module.load_state_dict(loaded_checkpoint["completion_state_dict"], strict=False)
    optimize_params = list(trainer.model.parameters())
    if trainer.mask_module is not None:
        optimize_params.extend(trainer.mask_module.parameters())
    if trainer.completion_module is not None:
        optimize_params.extend(trainer.completion_module.parameters())
    trainer.optimizer = torch.optim.AdamW(optimize_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    lr_gamma = float(args.lr_gamma)
    if abs(lr_gamma - 1.0) > 1e-12:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=lr_gamma)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_metric = float("inf")
    start_epoch = 0
    if bool(args.resume_from_checkpoint):
        if not isinstance(loaded_checkpoint, dict):
            raise ValueError("--resume_from_checkpoint requires a structured checkpoint")
        raw_history = loaded_checkpoint.get("history", [])
        if isinstance(raw_history, list):
            history = [dict(row) for row in raw_history if isinstance(row, dict)]
        for row in history:
            start_epoch = max(start_epoch, int(float(row.get("epoch", 0))))
        raw_best_metric = loaded_checkpoint.get("best_metric")
        if raw_best_metric is not None:
            best_metric = float(raw_best_metric)
        else:
            for row in history:
                metric = row.get("val_nmse", row.get("val_loss"))
                if metric is not None:
                    best_metric = min(best_metric, float(metric))
        optimizer_state = loaded_checkpoint.get("optimizer_state_dict")
        if bool(args.resume_optimizer_state) and optimizer_state is not None:
            trainer.optimizer.load_state_dict(optimizer_state)
        scheduler_state = loaded_checkpoint.get("scheduler_state_dict")
        if bool(args.resume_optimizer_state) and scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        print(
            "resume_from_checkpoint=True "
            f"start_epoch={start_epoch} target_epochs={int(args.epochs)} "
            f"history_rows={len(history)} best_metric={best_metric} "
            f"resume_optimizer_state={bool(args.resume_optimizer_state)}"
        )

    def save_checkpoint(
        history: list[dict[str, float]],
        *,
        filename: str = "strujepa_wifo_last.pt",
        best_metric: float | None = None,
    ) -> None:
        checkpoint = {
            "model_state_dict": trainer.model.state_dict(),
            "mask_module_state_dict": trainer.mask_module.state_dict() if trainer.mask_module is not None else None,
            "completion_state_dict": (
                trainer.completion_module.state_dict() if trainer.completion_module is not None else None
            ),
            "history": history,
            "args": vars(args),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        }
        if best_metric is not None:
            checkpoint["best_metric"] = float(best_metric)
        torch.save(checkpoint, save_dir / filename)
        with (save_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    if start_epoch >= int(args.epochs):
        print(
            "resume_noop "
            f"start_epoch={start_epoch} target_epochs={int(args.epochs)}"
        )
        save_checkpoint(history, best_metric=best_metric if best_metric < float("inf") else None)
        return

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        train_metrics = trainer.run_epoch(train_loader, epoch=epoch, train=True)
        record = {"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}}
        record["lr"] = float(trainer.optimizer.param_groups[0]["lr"])
        should_validate = (
            val_loader is not None
            and (epoch % max(1, int(args.validate_every)) == 0 or epoch == int(args.epochs))
        )
        if should_validate:
            with torch.inference_mode():
                val_metrics = trainer.run_epoch(val_loader, epoch=epoch, train=False)
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(record)
        metric = record.get("val_nmse", record.get("val_loss"))
        if metric is not None and float(metric) < best_metric:
            best_metric = float(metric)
            save_checkpoint(history, filename="strujepa_wifo_best.pt", best_metric=best_metric)
        if int(args.checkpoint_every) > 0 and (
            epoch % int(args.checkpoint_every) == 0 or epoch == int(args.epochs)
        ):
            save_checkpoint(history, best_metric=best_metric if best_metric < float("inf") else None)
        if scheduler is not None:
            scheduler.step()
        epoch = int(record["epoch"])
        train_loss = record.get("train_loss", float("nan"))
        train_nmse = record.get("train_nmse", float("nan"))
        val_loss = record.get("val_loss", float("nan"))
        val_nmse = record.get("val_nmse", float("nan"))
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_nmse={train_nmse:.6f} "
            f"val_loss={val_loss:.6f} val_nmse={val_nmse:.6f}"
        )
    save_checkpoint(history, best_metric=best_metric if best_metric < float("inf") else None)


if __name__ == "__main__":
    main()

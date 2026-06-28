#!/usr/bin/env python
"""Run StruJEPA on channel-estimation benchmark backbones."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method import StruJEPATrainer, ElasticizationSpec, MethodConfig  # noqa: E402
from elastic_method.run_paths import resolve_run_output_path  # noqa: E402
from integrations.channel_estimation.strujepa import (  # noqa: E402
    ADA_DATA_CONFIG,
    ADA_PAPER_STRICT_CONFIG,
    ADA_STATIC_CHECKPOINT,
    AMMSE_CHECKPOINT,
    AMMSE_CONFIG,
    build_adafortitran_strujepa,
    build_ammse_strujepa,
    build_channel_estimation_loaders,
    load_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/channel_estimation/current_adafortitran.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--val-limit", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-optimizer", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if not requested or requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def build_method_config(payload: dict[str, Any]) -> MethodConfig:
    return MethodConfig(
        supervised_weight=float(payload.get("supervised_weight", 1.0)),
        initialize_from_full_view=False,
        full_view_checkpoint=None,
        completion=payload.get("completion"),
    )


def build_spec(payload: dict[str, Any], *, family: str) -> ElasticizationSpec:
    return ElasticizationSpec(
        stack_path=str(payload.get("stack_path", "model")),
        block_family=family,
        width_multipliers=tuple(float(value) for value in payload.get("width_multipliers", [1.0, 0.75, 0.5])),
        depth_multipliers=tuple(float(value) for value in payload.get("depth_multipliers", [1.0, 0.5])),
        width_only_epochs=int(payload.get("width_only_epochs", 0)),
    )


def configure_optimizer(trainer: StruJEPATrainer, training_cfg: dict[str, Any]) -> None:
    params = list(trainer.model.parameters())
    if trainer.mask_module is not None:
        params.extend(trainer.mask_module.parameters())
    if trainer.completion_module is not None:
        params.extend(trainer.completion_module.parameters())
    trainer.optimizer = torch.optim.AdamW(
        params,
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    trainer: StruJEPATrainer,
    config: dict[str, Any],
    epoch: int,
    history: list[dict[str, float]],
    best_metric: float,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_state_dict": trainer.model.state_dict(),
            "mask_state_dict": None if trainer.mask_module is None else trainer.mask_module.state_dict(),
            "completion_state_dict": (
                None if trainer.completion_module is None else trainer.completion_module.state_dict()
            ),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": config,
            "history": history,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    model_cfg = config["model"]
    training_cfg = config["training"]
    method_cfg = config.get("method", {})
    elastic_cfg = config.get("elastic", {})

    if args.epochs > 0:
        training_cfg["epochs"] = args.epochs
    if args.batch_size > 0:
        training_cfg["batch_size"] = args.batch_size
    if args.device:
        training_cfg["device"] = args.device
    if args.output_dir:
        training_cfg["output_dir"] = args.output_dir

    model_family = str(model_cfg["family"]).strip().lower()
    checkpoint_raw = model_cfg.get("checkpoint_path")
    checkpoint_path = (
        Path(checkpoint_raw)
        if checkpoint_raw
        else (ADA_STATIC_CHECKPOINT if model_family == "adafortitran" else AMMSE_CHECKPOINT)
    )

    if model_family == "adafortitran":
        ada_config_raw = model_cfg.get("config_path")
        model, task_adapter = build_adafortitran_strujepa(
            variant=str(model_cfg.get("variant", "static_compat")),
            checkpoint_path=checkpoint_path,
            config_path=Path(ada_config_raw) if ada_config_raw else ADA_PAPER_STRICT_CONFIG,
        )
        loader_family = "adafortitran"
    elif model_family == "ammse":
        model, task_adapter = build_ammse_strujepa(
            checkpoint_path=checkpoint_path,
            config_path=model_cfg.get("config_path", AMMSE_CONFIG),
        )
        loader_family = "ammse"
    else:
        raise ValueError(f"unsupported model family '{model_family}'")

    metadata = {
        "model_family": model_family,
        "checkpoint_path": str(checkpoint_path),
        "metadata": vars(model.metadata),
        "training": training_cfg,
        "method": method_cfg,
        "elastic": elastic_cfg,
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0

    set_seed(int(training_cfg.get("seed", 13)))
    device = resolve_device(str(training_cfg.get("device", "auto")))
    output_dir = resolve_run_output_path(training_cfg["output_dir"], ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    train_limit = int(training_cfg.get("train_limit", 0)) if args.train_limit < 0 else int(args.train_limit)
    val_limit = int(training_cfg.get("val_limit", 0)) if args.val_limit < 0 else int(args.val_limit)
    train_loader, val_loader = build_channel_estimation_loaders(
        model_family=loader_family,
        data_config_path=config.get("data_config", ADA_DATA_CONFIG),
        batch_size=int(training_cfg.get("batch_size", 512)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        train_limit=train_limit,
        val_limit=val_limit,
        normalize_inputs=bool(training_cfg.get("normalize_inputs", True)),
    )

    trainer = StruJEPATrainer(
        model,
        task_adapter,
        spec=build_spec(elastic_cfg, family=str(model.metadata.family)),
        config=build_method_config(method_cfg),
        device=device,
    )
    configure_optimizer(trainer, training_cfg)

    epochs = int(training_cfg.get("epochs", 1))
    validate_every = int(training_cfg.get("validate_every", 1))
    best_metric = float("inf")
    history: list[dict[str, float]] = []
    start_epoch = 1
    if args.resume_checkpoint:
        checkpoint = torch.load(Path(args.resume_checkpoint), map_location="cpu", weights_only=False)
        missing, unexpected = trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"model resume mismatch: missing={missing}, unexpected={unexpected}")
        if trainer.mask_module is not None and checkpoint.get("mask_state_dict") is not None:
            trainer.mask_module.load_state_dict(checkpoint["mask_state_dict"], strict=True)
        if trainer.completion_module is not None and checkpoint.get("completion_state_dict") is not None:
            trainer.completion_module.load_state_dict(checkpoint["completion_state_dict"], strict=True)
        if args.resume_optimizer and checkpoint.get("optimizer_state_dict") is not None:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        history = list(checkpoint.get("history", []))

    scheduler = None
    lr_gamma = float(training_cfg.get("lr_gamma", 1.0))
    if abs(lr_gamma - 1.0) > 1e-12:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=lr_gamma)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = trainer.run_epoch(train_loader, epoch=epoch, train=True)
        record = {"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}}
        should_validate = epoch % max(1, validate_every) == 0 or epoch == epochs
        if should_validate:
            with torch.inference_mode():
                val_metrics = trainer.run_epoch(val_loader, epoch=epoch, train=False)
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            metric = float(val_metrics.get("nmse", val_metrics.get("loss", float("inf"))))
            if metric < best_metric:
                best_metric = metric
                save_checkpoint(
                    output_dir / "strujepa_best.pt",
                    trainer=trainer,
                    config=config,
                    epoch=epoch,
                    history=history,
                    best_metric=best_metric,
                )
        if scheduler is not None:
            scheduler.step()
        history.append(record)
        print(json.dumps(record), flush=True)

    save_checkpoint(
        output_dir / "strujepa_last.pt",
        trainer=trainer,
        config=config,
        epoch=epochs,
        history=history,
        best_metric=best_metric,
    )
    save_json(output_dir / "train_history.json", {"history": history, "best_metric": best_metric})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

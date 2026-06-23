from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method import ElasticizationSpec, MethodConfig, elasticize_model
from integrations.dinov3_imagenet.data import build_imagenet_loaders
from integrations.dinov3_imagenet.modeling import DINOv3ImageNetClassifier, build_linear_head, load_dinov3_backbone
from integrations.dinov3_imagenet.task import ImageNetClassificationTaskAdapter
from integrations.dinov3_imagenet.trainer import ClassificationStruJEPATrainer, ClassificationTrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StruJEPA on DINOv3 ImageNet classification.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dinov3_imagenet/current_dinov3.json"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    text = path.expanduser().resolve().read_text(encoding="utf-8")
    return json.loads(os.path.expandvars(text))


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(param_groups: list[dict[str, Any]], cfg_optim: dict[str, Any]) -> torch.optim.Optimizer:
    if cfg_optim.get("name", "adamw").lower() == "muon":
        try:
            from muon import Muon

            return Muon(
                param_groups,
                lr=cfg_optim["lr"],
                betas=tuple(cfg_optim.get("betas", (0.9, 0.95))),
                weight_decay=cfg_optim["weight_decay"],
            )
        except Exception:
            pass
    return torch.optim.AdamW(
        param_groups,
        lr=cfg_optim["lr"],
        betas=tuple(cfg_optim.get("betas", (0.9, 0.95))),
        weight_decay=cfg_optim["weight_decay"],
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    cfg_optim: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    sched_type = cfg_optim.get("sched", "cosine").lower()
    warmup_epochs = int(cfg_optim.get("warmup_epochs", 0))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        if sched_type == "cosine":
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    path.write_text(json.dumps(history, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    train_cfg = config["train"]
    optim_cfg = config["optim"]
    elastic_cfg = config["elastic"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    setup_seed(int(train_cfg.get("seed", 42)))
    out_dir = (args.output_dir or Path(train_cfg["out_dir"])).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] device={device} out_dir={out_dir}", flush=True)

    train_loader, val_loader, inferred_num_classes = build_imagenet_loaders(
        data_cfg["dataset_dir"],
        img_size=int(data_cfg.get("image_size", 224)),
        batch_size=int(data_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 8)),
    )
    print(
        f"[DATA] train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={data_cfg.get('batch_size', 16)}",
        flush=True,
    )

    backbone = load_dinov3_backbone(
        repo_dir=config["repo_dir"],
        backbone_name=model_cfg["backbone_name"],
        weight_path=model_cfg.get("backbone_weight_path"),
    )
    linear_head = build_linear_head(
        backbone=backbone,
        num_classes=int(model_cfg.get("num_classes", inferred_num_classes)),
        head_weight_path=model_cfg.get("head_weight_path"),
    )
    print(
        f"[MODEL] backbone={model_cfg['backbone_name']} embed_dim={backbone.embed_dim} "
        f"num_classes={int(model_cfg.get('num_classes', inferred_num_classes))}",
        flush=True,
    )
    classifier = DINOv3ImageNetClassifier(backbone, linear_head)

    elastic_model = elasticize_model(
        classifier,
        ElasticizationSpec(
            stack_path=str(elastic_cfg.get("stack_path", "backbone.blocks")),
            block_family=str(elastic_cfg.get("block_family", "dinov3_vit")),
            width_multipliers=tuple(float(value) for value in elastic_cfg["width_multipliers"]),
            depth_multipliers=tuple(float(value) for value in elastic_cfg["depth_multipliers"]),
            width_only_epochs=int(elastic_cfg.get("width_only_epochs", 0)),
        ),
    )
    param_dtype_name = str(train_cfg.get("param_dtype", "")).strip().lower()
    if device.type == "cuda" and param_dtype_name:
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if param_dtype_name not in dtype_map:
            raise ValueError(f"unsupported train.param_dtype: {train_cfg['param_dtype']!r}")
        elastic_model = elastic_model.to(dtype=dtype_map[param_dtype_name])

    method_config = MethodConfig(**config["method"])
    task_adapter = ImageNetClassificationTaskAdapter(
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )

    trainer = ClassificationStruJEPATrainer(
        elastic_model,
        task_adapter,
        spec=elastic_model.spec,
        method_config=method_config,
        trainer_config=ClassificationTrainerConfig(
            amp_dtype=str(train_cfg.get("amp_dtype", "bf16")),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            finetune_backbone=bool(train_cfg.get("finetune_backbone", True)),
        ),
        device=device,
        optimizer=None,
    )
    trainer.log_every_batches = int(train_cfg.get("log_every_batches", 0))

    param_groups: list[dict[str, Any]] = []
    if bool(train_cfg.get("finetune_backbone", True)):
        param_groups.append(
            {
                "params": elastic_model.model.backbone.parameters(),
                "lr": float(optim_cfg["lr"]) * float(train_cfg.get("backbone_lr_scale", 0.1)),
                "weight_decay": float(optim_cfg["weight_decay"]),
            }
        )
    param_groups.append(
        {
            "params": elastic_model.model.linear_head.parameters(),
            "lr": float(optim_cfg["lr"]),
            "weight_decay": float(optim_cfg["weight_decay"]),
        }
    )
    if trainer.completion_module is not None:
        param_groups.append(
            {
                "params": trainer.completion_module.parameters(),
                "lr": float(optim_cfg.get("completion_lr", optim_cfg["lr"])),
                "weight_decay": float(optim_cfg.get("completion_weight_decay", optim_cfg["weight_decay"])),
            }
        )

    optimizer = build_optimizer(param_groups, optim_cfg)
    scheduler = build_scheduler(optimizer, num_epochs=int(train_cfg["epochs"]), cfg_optim=optim_cfg)
    trainer.optimizer = optimizer
    print(
        f"[TRAIN] epochs={train_cfg['epochs']} "
        f"widths={elastic_cfg['width_multipliers']} depths={elastic_cfg['depth_multipliers']}",
        flush=True,
    )

    history: list[dict[str, float]] = []
    best_top1 = 0.0
    epochs = int(train_cfg["epochs"])
    eval_interval = int(train_cfg.get("eval_interval", 1))
    save_interval = int(train_cfg.get("save_interval", 1))

    for epoch in range(1, epochs + 1):
        print(f"[EPOCH] start {epoch}/{epochs}", flush=True)
        t0 = time.time()
        train_metrics = trainer.run_epoch(train_loader, epoch=epoch, train=True)
        scheduler.step()
        record: dict[str, float] = {"epoch": float(epoch)}
        record.update({f"train_{key}": value for key, value in train_metrics.items()})

        if epoch % eval_interval == 0:
            with torch.inference_mode():
                val_metrics = trainer.run_epoch(val_loader, epoch=epoch, train=False)
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            current_top1 = float(val_metrics.get("top1", float("nan")))
        else:
            current_top1 = float("nan")

        history.append(record)
        save_history(out_dir / "history.json", history)

        if epoch % save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "history": history,
                "best_top1": best_top1,
            }
            torch.save(checkpoint, out_dir / f"ckpt_epoch_{epoch:03d}.pt")

        if not math.isnan(current_top1) and current_top1 > best_top1:
            best_top1 = current_top1
            torch.save(
                {
                    "epoch": epoch,
                    "top1": best_top1,
                    "state_dict": trainer.model.model.linear_head.state_dict(),
                },
                out_dir / "linear_best.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "top1": best_top1,
                    "model_state_dict": trainer.model.state_dict(),
                    "config": config,
                },
                out_dir / "strujepa_best.pt",
            )

        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train/loss={train_metrics['loss']:.4f} "
            f"train/top1={train_metrics.get('top1', float('nan')):.2f} "
            f"train/top5={train_metrics.get('top5', float('nan')):.2f} "
            f"val/top1={record.get('val_top1', float('nan')):.2f} "
            f"val/top5={record.get('val_top5', float('nan')):.2f} "
            f"({elapsed:.1f}s)"
        )

    print(f"[DONE] Best val top1: {best_top1:.2f}")


if __name__ == "__main__":
    main()

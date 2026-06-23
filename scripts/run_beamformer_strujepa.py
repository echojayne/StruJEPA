#!/usr/bin/env python
"""Train StruJEPA elastic subnets for BeamFormer's estimator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method import MethodConfig, StruJEPATrainer
from integrations.beamformer.strujepa import (
    BeamFormerTaskAdapter,
    build_loaders,
    build_setting,
    elasticize_beamformer_estimator,
    load_original_estimator,
    load_original_generator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "beamformer" / "current_beamformer.json")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--val-limit", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-weights-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if not raw or raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(raw)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = dict(config.get("training", {}))
    data = dict(config.get("data", {}))
    elastic = dict(config["elastic"])
    depth_selection = str(elastic.get("depth_selection", "")).strip()
    if depth_selection:
        os.environ["STRUJEPA_DEPTH_SELECTION"] = depth_selection

    if args.epochs > 0:
        training["epochs"] = args.epochs
    if args.batch_size > 0:
        training["batch_size"] = args.batch_size
    if args.train_limit >= 0:
        training["train_limit"] = args.train_limit
    if args.val_limit >= 0:
        training["val_limit"] = args.val_limit
    if args.output_dir is not None:
        training["output_dir"] = str(args.output_dir)
    if args.device:
        training["device"] = args.device
    config["training"] = training

    set_seed(int(training.get("seed", 13)))
    device = resolve_device(str(training.get("device", "auto")))
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model_cfg = dict(config.get("model", {}))
    setting_kwargs = {}
    if data.get("csi_root"):
        setting_kwargs["csi_root"] = data.get("csi_root")
    if model_cfg.get("weight_root"):
        setting_kwargs["weight_root"] = model_cfg.get("weight_root")
    if model_cfg.get("source_root"):
        setting_kwargs["source_root"] = model_cfg.get("source_root")
    setting = build_setting(**setting_kwargs)
    train_loader, val_loader = build_loaders(
        setting,
        batch_size=int(training.get("batch_size", 2)),
        num_workers=int(training.get("num_workers", 0)),
        train_split=str(data.get("train_split", "t16x16_r2x1_train")),
        val_split=str(data.get("val_split", "t16x16_r2x1_test_small")),
        train_limit=int(training.get("train_limit", 0)),
        val_limit=int(training.get("val_limit", 0)),
    )
    estimator = load_original_estimator(setting)
    model, spec = elasticize_beamformer_estimator(
        estimator,
        width_multipliers=tuple(float(value) for value in elastic["width_multipliers"]),
        depth_multipliers=tuple(float(value) for value in elastic["depth_multipliers"]),
        copy_model=False,
    )
    generator = load_original_generator(setting, device)
    task_adapter = BeamFormerTaskAdapter(
        setting,
        generator=generator,
        device=device,
        use_lpe=bool(data.get("use_lpe", True)),
    )
    method_config = MethodConfig(**dict(config.get("method", {})))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-5)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    trainer = StruJEPATrainer(
        model,
        task_adapter,
        spec=spec,
        config=method_config,
        device=device,
        optimizer=optimizer,
    )
    trainer.log_every_batches = int(training.get("log_every_batches", 0))
    initial_history: list[dict[str, Any]] = []
    initial_stage_reports: list[dict[str, Any]] = []
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError("resume checkpoint must contain a dictionary payload")
        elastic_state = checkpoint.get("elastic_state_dict")
        if not isinstance(elastic_state, dict):
            raise KeyError("resume checkpoint does not contain elastic_state_dict")
        model.load_state_dict(elastic_state, strict=True)
        completion_state = checkpoint.get("completion_state_dict")
        if completion_state is not None:
            if trainer.completion_module is None:
                raise RuntimeError("resume checkpoint contains completion state but completion is disabled")
            trainer.completion_module.load_state_dict(completion_state, strict=True)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if not args.resume_weights_only and isinstance(optimizer_state, dict):
            trainer.optimizer.load_state_dict(optimizer_state)
        configured_lr = float(training.get("learning_rate", 1e-5))
        for group in trainer.optimizer.param_groups:
            group["lr"] = configured_lr
        if bool(method_config.reference_from_resume):
            trainer.refresh_reference_from_student()
        if not args.resume_weights_only:
            initial_history = list(checkpoint.get("history", []))
            initial_stage_reports = list(checkpoint.get("stage_reports", []))
        print(
            json.dumps(
                {
                    "event": "resume_loaded",
                    "checkpoint": str(args.resume_checkpoint),
                    "weights_only": bool(args.resume_weights_only),
                    "history_records": len(initial_history),
                    "stage_reports": len(initial_stage_reports),
                    "learning_rate": configured_lr,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    def checkpoint_payload(
        history_snapshot: list[dict[str, Any]],
        stage_reports_snapshot: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "model_state_dict": model.model.state_dict(),
            "elastic_state_dict": model.state_dict(),
            "completion_state_dict": (
                trainer.completion_module.state_dict()
                if trainer.completion_module is not None
                else None
            ),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": config,
            "history": history_snapshot,
            "stage_reports": stage_reports_snapshot,
            "metadata": {
                "family": model.metadata.family,
                "total_layers": model.metadata.total_layers,
                "max_num_heads": model.metadata.max_num_heads,
                "max_ffn_dim": model.metadata.max_ffn_dim,
            },
        }

    def save_progress(
        history_snapshot: list[dict[str, Any]],
        stage_reports_snapshot: list[dict[str, Any]],
    ) -> None:
        history_for_checkpoint = list(history_snapshot)
        reports_for_checkpoint = list(stage_reports_snapshot)
        payload = checkpoint_payload(history_for_checkpoint, reports_for_checkpoint)
        (output_dir / "train_history.partial.json").write_text(
            json.dumps(history_snapshot, indent=2),
            encoding="utf-8",
        )
        if stage_reports_snapshot:
            (output_dir / "stage_reports.partial.json").write_text(
                json.dumps(stage_reports_snapshot, indent=2),
                encoding="utf-8",
            )
        torch.save(payload, output_dir / "beamformer_strujepa_progress.pt")
        last_record = history_for_checkpoint[-1] if history_for_checkpoint else {}
        if bool(last_record.get("convergence_improved", False)):
            stage = str(last_record.get("stage", "stage"))
            torch.save(payload, output_dir / "beamformer_strujepa_best.pt")
            torch.save(payload, output_dir / f"beamformer_strujepa_best_{stage}.pt")

    stage_reports: list[dict[str, Any]] = []
    if bool((method_config.stage_convergence or {}).get("enabled", False)):
        history, stage_reports = trainer.fit_until_converged(
            train_loader,
            val_loader=val_loader,
            on_epoch_end=save_progress,
            initial_history=initial_history,
            initial_stage_reports=initial_stage_reports,
        )
    else:
        history = trainer.fit(train_loader, epochs=int(training.get("epochs", 1)), val_loader=val_loader)

    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if stage_reports:
        (output_dir / "stage_reports.json").write_text(
            json.dumps(stage_reports, indent=2),
            encoding="utf-8",
        )

    payload = checkpoint_payload(history, stage_reports)
    torch.save(payload, output_dir / "beamformer_strujepa_last.pt")
    torch.save(payload, output_dir / "beamformer_strujepa_best.pt")
    final_report = {"last": history[-1] if history else {}, "stage_reports": stage_reports}
    print(json.dumps(final_report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""StruJEPA integration helpers for traffic-forecasting benchmarks."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from elastic_method import ElasticizationSpec, elasticize_model
from elastic_method.core.structures import ForwardResult


BENCHMARK_ROOT = Path(
    os.environ.get("AI_RAN_BENCHMARK_ROOT", Path.home() / "ai_ran_benchmarks")
).expanduser()
os.environ.setdefault("AI_RAN_BENCHMARK_ROOT", str(BENCHMARK_ROOT))
SOURCE_PARENT = BENCHMARK_ROOT / "source_snapshots"
ITRANSFORMER_CONFIG = SOURCE_PARENT / "traffic_forecasting/config/milan_itransformer_train.yaml"
ITRANSFORMER_CHECKPOINT = (
    BENCHMARK_ROOT / "benchmarks/traffic_forecasting/itransformer/assets/static_baseline/best.pt"
)


def ensure_benchmark_source_path() -> None:
    source_parent = str(SOURCE_PARENT)
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(os.path.expandvars(text))


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        state_dict = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(state_dict, dict):
            if state_dict and all(name.startswith("model.") for name in state_dict):
                return {name.removeprefix("model."): value for name, value in state_dict.items()}
            return state_dict
    raise KeyError(f"unsupported checkpoint format: {path}")


class TrafficForecastingTaskAdapter:
    """Task adapter for normalized sequence forecasting windows."""

    def prepare_batch(self, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
        return {"model_args": (inputs,), "model_kwargs": {}, "targets": targets}

    def batch_size(self, batch: dict[str, Any]) -> int:
        return int(batch["targets"].shape[0])

    @staticmethod
    def _prediction(result: ForwardResult) -> torch.Tensor:
        output = result.model_output
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("traffic-forecasting task adapter expects tensor predictions")
        return output

    def compute_supervised_loss(self, result: ForwardResult, batch: dict[str, Any]) -> torch.Tensor:
        prediction = self._prediction(result)
        return torch.mean((prediction - batch["targets"]) ** 2)

    def compute_metrics(self, result: ForwardResult, batch: dict[str, Any]) -> dict[str, float]:
        prediction = self._prediction(result)
        diff = prediction - batch["targets"]
        mse = float(torch.mean(diff**2).detach().item())
        mae = float(torch.mean(torch.abs(diff)).detach().item())
        rmse = float(torch.sqrt(torch.mean(diff**2)).detach().item())
        return {"mse": mse, "mae": mae, "rmse": rmse}


def build_itransformer_strujepa(
    *,
    config_path: str | Path = ITRANSFORMER_CONFIG,
    checkpoint_path: str | Path = ITRANSFORMER_CHECKPOINT,
    strict_load: bool = True,
):
    """Build the benchmark static iTransformer as an elastic StruJEPA model."""

    ensure_benchmark_source_path()
    from traffic_forecasting.data import load_manifest  # noqa: WPS433
    from traffic_forecasting.models import ITransformerConfig, ITransformerModel  # noqa: WPS433

    config = load_yaml(config_path)
    manifest = load_manifest(config)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model_cfg = config.get("model", {})
    if isinstance(checkpoint_config, dict):
        model_cfg = {**model_cfg, **checkpoint_config.get("model", {})}
    model = ITransformerModel(
        ITransformerConfig(
            num_regions=int(manifest["top_k_regions"]),
            context_len=int(manifest["context_len"]),
            horizon=int(manifest["horizon"]),
            d_model=int(model_cfg.get("d_model", 128)),
            depth=int(model_cfg.get("depth", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=int(model_cfg.get("ffn_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    )
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    spec = ElasticizationSpec(
        stack_path="encoder",
        block_family="torch_encoder",
        width_multipliers=(1.0,),
        depth_multipliers=(1.0,),
    )
    elastic_model = elasticize_model(model, spec, copy_model=False)
    task_adapter = TrafficForecastingTaskAdapter()
    return elastic_model, task_adapter


def build_itransformer_loaders(
    *,
    config_path: str | Path = ITRANSFORMER_CONFIG,
    batch_size: int,
    num_workers: int,
    train_limit: int = 0,
    val_limit: int = 0,
):
    ensure_benchmark_source_path()
    from traffic_forecasting.data import build_split_dataloader  # noqa: WPS433

    config = load_yaml(config_path)
    train_loader = build_split_dataloader(
        config,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        limit=train_limit,
        normalize=True,
    )
    val_loader = build_split_dataloader(
        config,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        limit=val_limit,
        normalize=True,
    )
    return train_loader, val_loader

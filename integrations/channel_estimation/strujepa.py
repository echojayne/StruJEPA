"""StruJEPA integration helpers for channel-estimation benchmarks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from elastic_method import ElasticizationSpec, elasticize_model
from elastic_method.core.multi_wrapper import MultiStackTorchEncoderWrapper
from elastic_method.core.structures import ForwardResult


BENCHMARK_ROOT = Path(
    os.environ.get("AI_RAN_BENCHMARK_ROOT", Path.home() / "ai_ran_benchmarks")
).expanduser()
os.environ.setdefault("AI_RAN_BENCHMARK_ROOT", str(BENCHMARK_ROOT))
SOURCE_PARENT = BENCHMARK_ROOT / "source_snapshots"
CHANNEL_SOURCE_ROOT = SOURCE_PARENT / "channel_estimation"
ADA_PUBLIC_SOURCE_ROOT = CHANNEL_SOURCE_ROOT / "upstream_public/adafortitran_public"
ADA_STATIC_CHECKPOINT = (
    BENCHMARK_ROOT
    / "benchmarks/channel_estimation/adafortitran/assets/static_baseline/seed13_lr1e-3/best.pt"
)
ADA_PAPER_STRICT_CONFIG = SOURCE_PARENT / "channel_estimation/config/adafortitran_paper_strict_repro_train.yaml"
ADA_DATA_CONFIG = SOURCE_PARENT / "channel_estimation/config/adafortitran_reference_benchmark.yaml"
AMMSE_CHECKPOINT = (
    BENCHMARK_ROOT
    / "benchmarks/channel_estimation/ammse/assets/paper_strict_current_benchmark_final/best.pt"
)
AMMSE_CONFIG = SOURCE_PARENT / "channel_estimation/config/ammse_paper_strict_current_benchmark.yaml"


def ensure_benchmark_source_path() -> None:
    for path in (ADA_PUBLIC_SOURCE_ROOT, CHANNEL_SOURCE_ROOT, SOURCE_PARENT):
        if not path.exists():
            continue
        source_path = str(path)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(os.path.expandvars(text))


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        state_dict = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(f"unsupported checkpoint format: {path}")


def _convert_public_adafortitran_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map official public AdaFortiTran checkpoint keys to the local backbone names."""

    if not any(key.startswith("pilot_upsampler.") for key in state_dict):
        return state_dict

    replacements = (
        ("pilot_upsampler.", "upsampler.linear."),
        ("initial_enhancer.conv_block.", "initial_enhancer.net."),
        ("transformer_encoder.linear_1.", "transformer_encoder.input_proj."),
        ("transformer_encoder.positional_encoding.", "transformer_encoder.position."),
        ("transformer_encoder.transformer.layers.", "transformer_encoder.layers."),
        ("transformer_encoder.linear_2.", "transformer_encoder.output_proj."),
        ("final_refiner.conv_block.", "final_refiner.net."),
    )
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        converted_key = key.removeprefix("module.")
        for old, new in replacements:
            if converted_key.startswith(old):
                converted_key = new + converted_key[len(old) :]
                break
        converted[converted_key] = value
    return converted


def metadata_conditioning(
    metadata_list: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not metadata_list:
        return None
    snr = torch.tensor([float(item["snr"]) for item in metadata_list], device=device, dtype=torch.float32)
    delay_spread = torch.tensor(
        [float(item["delay_spread"]) for item in metadata_list],
        device=device,
        dtype=torch.float32,
    )
    doppler = torch.tensor([float(item["doppler"]) for item in metadata_list], device=device, dtype=torch.float32)
    return snr, delay_spread, doppler


def batch_nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, prediction.ndim))
    numerator = torch.sum((prediction - target) ** 2, dim=dims)
    denominator = torch.sum(target**2, dim=dims).clamp_min(1e-12)
    return torch.mean(numerator / denominator)


@dataclass
class DenseChannelEstimationTaskAdapter:
    """Task adapter for dense OFDM channel field regression."""

    model_family: str
    target_key: str
    use_conditioning: bool = False
    use_noise_var: bool = False

    def prepare_batch(self, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        pilot_vector = batch["pilot_vector"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch[self.target_key].to(device=device, dtype=torch.float32, non_blocking=True)
        model_kwargs: dict[str, Any] = {}
        model_args: tuple[Any, ...]
        family = self.model_family.strip().lower()
        if family == "ammse":
            model_args = (pilot_vector,)
            model_kwargs["return_representation"] = True
            if self.use_noise_var:
                model_kwargs["noise_var"] = batch["noise_var"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
        elif family == "adafortitran":
            model_args = ()
            model_kwargs["pilot_vector"] = pilot_vector
            if self.use_conditioning:
                model_kwargs["conditioning"] = metadata_conditioning(batch.get("metadata", []), device=device)
        else:
            raise ValueError(f"unsupported channel-estimation model family '{self.model_family}'")
        return {"model_args": model_args, "model_kwargs": model_kwargs, "targets": targets}

    def batch_size(self, batch: dict[str, Any]) -> int:
        return int(batch["targets"].shape[0])

    @staticmethod
    def _prediction(result: ForwardResult) -> torch.Tensor:
        output = result.model_output
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("channel-estimation task adapter expects tensor predictions")
        return output

    def compute_supervised_loss(self, result: ForwardResult, batch: dict[str, Any]) -> torch.Tensor:
        prediction = self._prediction(result)
        return torch.mean((prediction - batch["targets"]) ** 2)

    def compute_metrics(self, result: ForwardResult, batch: dict[str, Any]) -> dict[str, float]:
        prediction = self._prediction(result)
        target = batch["targets"]
        diff = prediction - target
        mse = float(torch.mean(diff**2).detach().item())
        mae = float(torch.mean(torch.abs(diff)).detach().item())
        nmse = float(batch_nmse(prediction, target).detach().item())
        return {"mse": mse, "mae": mae, "nmse": nmse}


def _ada_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    use_channel_adaptation = bool(model_cfg.get("use_channel_adaptation", False))
    return {
        "num_subcarriers": int(model_cfg.get("num_subcarriers", 120)),
        "num_symbols": int(model_cfg.get("num_symbols", 14)),
        "input_channels": int(model_cfg.get("input_channels", 4)),
        "output_channels": int(model_cfg.get("output_channels", 2)),
        "pilot_vector_length": int(model_cfg.get("pilot_vector_length", 80)),
        "d_enc": int(model_cfg.get("d_enc", 32)),
        "encoder_layers": int(model_cfg.get("encoder_layers", 6)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "ffn_dim": int(model_cfg.get("ffn_dim", 64)),
        "activation": str(model_cfg.get("activation", "gelu")),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "max_seq_len": int(model_cfg.get("max_seq_len", 512)),
        "pos_encoding_type": str(model_cfg.get("pos_encoding_type", "learnable")),
        "patch_subcarriers": int(model_cfg.get("patch_subcarriers", 3)),
        "patch_symbols": int(model_cfg.get("patch_symbols", 2)),
        "shallow_channels": int(model_cfg.get("shallow_channels", 8)),
        "hidden_channels": int(model_cfg.get("hidden_channels", 32)),
        "use_channel_adaptation": use_channel_adaptation,
        "channel_adaptivity_hidden_sizes": tuple(
            int(value) for value in model_cfg.get("channel_adaptivity_hidden_sizes", [7, 42, 560])
        )
        if use_channel_adaptation
        else None,
        "adaptive_token_length": (
            int(model_cfg["adaptive_token_length"])
            if model_cfg.get("adaptive_token_length") is not None
            else None
        ),
    }


def build_adafortitran_strujepa(
    *,
    variant: str = "static_compat",
    checkpoint_path: str | Path = ADA_STATIC_CHECKPOINT,
    config_path: str | Path = ADA_PAPER_STRICT_CONFIG,
    strict_load: bool = True,
):
    """Build an elastic AdaFortiTran/FortiTran-compatible StruJEPA model."""

    ensure_benchmark_source_path()
    from channel_estimation.models.adafortitran import (  # noqa: WPS433
        AdaFortiTranBackbone,
        AdaFortiTranConfig,
        LegacyAdaFortiTranStaticCompat,
    )

    variant_key = variant.strip().lower()
    if variant_key == "static_compat":
        model = LegacyAdaFortiTranStaticCompat()
        stack_path = "encoder"
        use_conditioning = False
    elif variant_key in {"paper_strict", "adafortitran"}:
        train_cfg = load_yaml(config_path)
        model_config = _ada_model_config(train_cfg["model"])
        model = AdaFortiTranBackbone(AdaFortiTranConfig(**model_config))
        stack_path = "transformer_encoder.layers"
        use_conditioning = bool(model_config["use_channel_adaptation"])
    else:
        raise ValueError(f"unsupported AdaFortiTran variant '{variant}'")

    state_dict = _convert_public_adafortitran_state_dict(load_checkpoint_state_dict(checkpoint_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    spec = ElasticizationSpec(
        stack_path=stack_path,
        block_family="torch_encoder",
        width_multipliers=(1.0,),
        depth_multipliers=(1.0,),
    )
    elastic_model = elasticize_model(model, spec, copy_model=False)
    task_adapter = DenseChannelEstimationTaskAdapter(
        model_family="adafortitran",
        target_key="target",
        use_conditioning=use_conditioning,
    )
    return elastic_model, task_adapter


def build_ammse_strujepa(
    *,
    checkpoint_path: str | Path = AMMSE_CHECKPOINT,
    config_path: str | Path = AMMSE_CONFIG,
    strict_load: bool = True,
):
    """Build an elastic rank-adaptive A-MMSE StruJEPA model."""

    ensure_benchmark_source_path()
    from channel_estimation.models.ammse_rank_adaptive import (  # noqa: WPS433
        AMMSERankAdaptiveConfig,
        AMMSERankAdaptiveModel,
    )

    train_cfg = load_yaml(config_path)
    model_cfg = dict(train_cfg["model"])
    model_cfg.pop("architecture", None)
    model = AMMSERankAdaptiveModel(AMMSERankAdaptiveConfig(**model_cfg))
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    elastic_model = MultiStackTorchEncoderWrapper(
        model,
        stack_paths=("encoder.frequency_encoder", "encoder.temporal_encoder"),
    )
    task_adapter = DenseChannelEstimationTaskAdapter(
        model_family="ammse",
        target_key="target_full_grid",
        use_noise_var=True,
    )
    return elastic_model, task_adapter


def build_channel_estimation_loaders(
    *,
    model_family: str,
    data_config_path: str | Path = ADA_DATA_CONFIG,
    batch_size: int,
    num_workers: int,
    train_limit: int = 0,
    val_limit: int = 0,
    normalize_inputs: bool = True,
):
    ensure_benchmark_source_path()
    data_config = load_yaml(data_config_path)
    family = model_family.strip().lower()
    if family == "adafortitran":
        from channel_estimation.data import build_split_dataloader  # noqa: WPS433

        train_loader = build_split_dataloader(
            data_config,
            split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            limit=train_limit,
            normalize_inputs=normalize_inputs,
        )
        val_loader = build_split_dataloader(
            data_config,
            split="val",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            limit=val_limit,
            normalize_inputs=normalize_inputs,
        )
        return train_loader, val_loader
    if family == "ammse":
        from channel_estimation.data import build_ammse_dataloader  # noqa: WPS433

        train_loader = build_ammse_dataloader(
            data_config,
            split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            limit=train_limit,
        )
        val_loader = build_ammse_dataloader(
            data_config,
            split="val",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            limit=val_limit,
        )
        return train_loader, val_loader
    raise ValueError(f"unsupported channel-estimation model family '{model_family}'")

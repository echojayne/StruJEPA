from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from thop import profile

from elastic_wifo import elasticize_wifo
from elastic_method.core.subnet import resolve_active_ffn, resolve_active_heads, resolve_active_layers
from model import WiFo_model
from strujepa_data import load_channel_tensor
from strujepa_wifo import parse_multiplier_string


ROOT = Path(__file__).resolve().parents[2]
RUN_DEFAULT = (
    ROOT
    / "runs"
    / "wifo_base_strujepa_gpu0_anchor_random_20260412_223720"
)
WEIGHTS_DIR = ROOT / "WIFO" / "weights"
DATASET_DIR = ROOT / "WIFO" / "dataset"
PLOT_DATASETS = ("D1", "D5", "D9", "D15")
FULL_DATASETS = tuple(f"D{i}" for i in range(1, 17))
RAW_MODEL_SIZES = ("tiny", "little", "small", "base", "large")
PLOT_RAW_MODEL_SIZES = ("tiny", "little", "small", "base")
DEFAULT_SUBNET_WIDTHS = (1.0, 0.5, 0.125)
DEFAULT_SUBNET_DEPTHS = (1.0, 0.5, 0.166667)
TASKS = (("temporal", "Temporal"), ("fre", "Frequency"))
PAPER_TIME_BASELINES = {
    "D1": {"WiFo-Base": 0.082, "Transformer": 0.112, "LSTM": 0.356, "3D ResNet": 0.088, "PAD": 0.529, "LLM4CP": 0.117, "LLM4CP*": 0.074},
    "D2": {"WiFo-Base": 0.260, "Transformer": 0.416, "LSTM": 0.797, "3D ResNet": 0.351, "PAD": 1.074, "LLM4CP": 0.451, "LLM4CP*": 0.305},
    "D3": {"WiFo-Base": 0.016, "Transformer": 0.016, "LSTM": 0.027, "3D ResNet": 0.014, "PAD": 0.038, "LLM4CP": 0.015, "LLM4CP*": 0.013},
    "D4": {"WiFo-Base": 0.048, "Transformer": 0.107, "LSTM": 0.418, "3D ResNet": 0.055, "PAD": 0.317, "LLM4CP": 0.106, "LLM4CP*": 0.060},
    "D5": {"WiFo-Base": 0.494, "Transformer": 0.638, "LSTM": 0.788, "3D ResNet": 0.751, "PAD": 5.008, "LLM4CP": 0.637, "LLM4CP*": 0.510},
    "D6": {"WiFo-Base": 0.095, "Transformer": 0.174, "LSTM": 0.542, "3D ResNet": 0.157, "PAD": 0.568, "LLM4CP": 0.206, "LLM4CP*": 0.133},
    "D7": {"WiFo-Base": 0.081, "Transformer": 0.219, "LSTM": 0.576, "3D ResNet": 0.103, "PAD": 0.617, "LLM4CP": 0.198, "LLM4CP*": 0.112},
    "D8": {"WiFo-Base": 0.018, "Transformer": 0.024, "LSTM": 0.092, "3D ResNet": 0.016, "PAD": 0.073, "LLM4CP": 0.025, "LLM4CP*": 0.016},
    "D9": {"WiFo-Base": 0.347, "Transformer": 0.483, "LSTM": 0.835, "3D ResNet": 0.349, "PAD": 1.087, "LLM4CP": 0.475, "LLM4CP*": 0.312},
    "D10": {"WiFo-Base": 0.467, "Transformer": 0.649, "LSTM": 0.689, "3D ResNet": 0.869, "PAD": 3.863, "LLM4CP": 0.709, "LLM4CP*": 0.563},
    "D11": {"WiFo-Base": 0.227, "Transformer": 0.440, "LSTM": 0.834, "3D ResNet": 0.274, "PAD": 1.017, "LLM4CP": 0.405, "LLM4CP*": 0.273},
    "D12": {"WiFo-Base": 0.023, "Transformer": 0.035, "LSTM": 0.166, "3D ResNet": 0.025, "PAD": 0.132, "LLM4CP": 0.035, "LLM4CP*": 0.026},
    "D13": {"WiFo-Base": 0.482, "Transformer": 0.718, "LSTM": 0.876, "3D ResNet": 0.815, "PAD": 5.213, "LLM4CP": 0.758, "LLM4CP*": 0.648},
    "D14": {"WiFo-Base": 0.369, "Transformer": 0.546, "LSTM": 0.884, "3D ResNet": 0.388, "PAD": 1.021, "LLM4CP": 0.562, "LLM4CP*": 0.358},
    "D15": {"WiFo-Base": 0.029, "Transformer": 0.039, "LSTM": 0.156, "3D ResNet": 0.032, "PAD": 0.151, "LLM4CP": 0.038, "LLM4CP*": 0.030},
    "D16": {"WiFo-Base": 0.318, "Transformer": 0.591, "LSTM": 0.944, "3D ResNet": 0.329, "PAD": 1.034, "LLM4CP": 0.545, "LLM4CP*": 0.349},
    "Average": {"WiFo-Base": 0.210, "Transformer": 0.325, "LSTM": 0.561, "3D ResNet": 0.289, "PAD": 1.359, "LLM4CP": 0.330, "LLM4CP*": 0.236},
}
PAPER_FREQUENCY_BASELINES = {
    "D1": {"WiFo-Base": 0.318, "Transformer": 0.532, "LSTM": 0.705, "3D ResNet": 0.839, "LLM4CP": 0.392, "LLM4CP*": 0.375},
    "D2": {"WiFo-Base": 0.181, "Transformer": 0.556, "LSTM": 0.763, "3D ResNet": 0.647, "LLM4CP": 0.419, "LLM4CP*": 0.223},
    "D3": {"WiFo-Base": 0.027, "Transformer": 0.016, "LSTM": 0.037, "3D ResNet": 0.071, "LLM4CP": 0.023, "LLM4CP*": 0.025},
    "D4": {"WiFo-Base": 0.073, "Transformer": 0.270, "LSTM": 0.475, "3D ResNet": 0.215, "LLM4CP": 0.211, "LLM4CP*": 0.151},
    "D5": {"WiFo-Base": 0.152, "Transformer": 0.315, "LSTM": 0.577, "3D ResNet": 0.386, "LLM4CP": 0.267, "LLM4CP*": 0.165},
    "D6": {"WiFo-Base": 0.081, "Transformer": 0.310, "LSTM": 0.540, "3D ResNet": 0.458, "LLM4CP": 0.193, "LLM4CP*": 0.140},
    "D7": {"WiFo-Base": 0.092, "Transformer": 0.392, "LSTM": 0.578, "3D ResNet": 0.354, "LLM4CP": 0.318, "LLM4CP*": 0.189},
    "D8": {"WiFo-Base": 0.061, "Transformer": 0.024, "LSTM": 0.348, "3D ResNet": 0.139, "LLM4CP": 0.068, "LLM4CP*": 0.069},
    "D9": {"WiFo-Base": 0.436, "Transformer": 0.481, "LSTM": 0.895, "3D ResNet": 0.918, "LLM4CP": 0.574, "LLM4CP*": 0.418},
    "D10": {"WiFo-Base": 0.087, "Transformer": 0.261, "LSTM": 0.451, "3D ResNet": 0.257, "LLM4CP": 0.163, "LLM4CP*": 0.096},
    "D11": {"WiFo-Base": 0.245, "Transformer": 0.723, "LSTM": 0.859, "3D ResNet": 0.823, "LLM4CP": 0.621, "LLM4CP*": 0.349},
    "D12": {"WiFo-Base": 0.023, "Transformer": 0.048, "LSTM": 0.131, "3D ResNet": 0.029, "LLM4CP": 0.032, "LLM4CP*": 0.026},
    "D13": {"WiFo-Base": 0.068, "Transformer": 0.238, "LSTM": 0.531, "3D ResNet": 0.177, "LLM4CP": 0.165, "LLM4CP*": 0.067},
    "D14": {"WiFo-Base": 0.395, "Transformer": 0.744, "LSTM": 0.911, "3D ResNet": 0.924, "LLM4CP": 0.637, "LLM4CP*": 0.414},
    "D15": {"WiFo-Base": 0.023, "Transformer": 0.053, "LSTM": 0.083, "3D ResNet": 0.045, "LLM4CP": 0.024, "LLM4CP*": 0.024},
    "D16": {"WiFo-Base": 0.270, "Transformer": 0.855, "LSTM": 0.929, "3D ResNet": 0.723, "LLM4CP": 0.712, "LLM4CP*": 0.456},
    "Average": {"WiFo-Base": 0.158, "Transformer": 0.364, "LSTM": 0.551, "3D ResNet": 0.438, "LLM4CP": 0.301, "LLM4CP*": 0.199},
}


def quiet_build_model(args: SimpleNamespace) -> torch.nn.Module:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = WiFo_model(args=args)
    return model


def build_wifo_args(size: str) -> SimpleNamespace:
    return SimpleNamespace(
        size=size,
        t_patch_size=4,
        patch_size=4,
        pos_emb="SinCos_3D",
        no_qkv_bias=0,
    )


def weight_path_for_size(size: str) -> Path:
    return WEIGHTS_DIR / f"wifo_{size}.pkl"


def load_raw_model(size: str, device: torch.device) -> torch.nn.Module:
    args = build_wifo_args(size)
    model = quiet_build_model(args).to(device)
    state_dict = torch.load(weight_path_for_size(size), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_strujepa_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    width_multipliers: tuple[float, ...],
    depth_multipliers: tuple[float, ...],
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    base_model = quiet_build_model(args)
    elastic_model = elasticize_wifo(
        base_model,
        width_multipliers=width_multipliers,
        depth_multipliers=depth_multipliers,
        width_only_epochs=int(getattr(args, "width_only_epochs", 0)),
        copy_model=True,
    )
    elastic_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    elastic_model.to(device)
    elastic_model.eval()
    return elastic_model


def d1_kept_tokens(mask_strategy: str) -> int:
    total_t = 24 // 4
    total_h = 4 // 4
    total_w = 128 // 4
    if mask_strategy == "temporal":
        return int(total_t * 0.5) * total_h * total_w
    if mask_strategy == "fre":
        return total_t * total_h * int(total_w * 0.5)
    return int(total_t * total_h * total_w * 0.5)


def encoder_macs_per_sample(
    *,
    seq_len: int,
    embed_dim: int,
    active_dim: int,
    active_ffn_dim: int,
    active_layers: int,
) -> float:
    qkv_macs = 3 * seq_len * embed_dim * active_dim
    attn_macs = 2 * seq_len * seq_len * active_dim
    proj_macs = seq_len * active_dim * embed_dim
    ffn_macs = 2 * seq_len * embed_dim * active_ffn_dim
    return float(active_layers * (qkv_macs + attn_macs + proj_macs + ffn_macs))


def estimate_raw_encoder_macs(model: torch.nn.Module, *, mask_strategy: str) -> float:
    return encoder_macs_per_sample(
        seq_len=d1_kept_tokens(mask_strategy),
        embed_dim=int(model.embed_dim),
        active_dim=int(model.embed_dim),
        active_ffn_dim=int(model.blocks[0].mlp.fc1.out_features),
        active_layers=int(model.depth),
    )


def estimate_elastic_encoder_macs(
    elastic_model: torch.nn.Module,
    *,
    mask_strategy: str,
    width_multiplier: float,
    depth_multiplier: float,
) -> float:
    metadata = elastic_model.metadata
    head_dim = int(elastic_model.model.embed_dim // metadata.max_num_heads)
    active_heads = resolve_active_heads(
        max_heads=metadata.max_num_heads,
        width_multiplier=width_multiplier,
    )
    active_ffn_dim = resolve_active_ffn(
        max_ffn_dim=metadata.max_ffn_dim,
        width_multiplier=width_multiplier,
    )
    active_layers = resolve_active_layers(
        max_layers=metadata.total_layers,
        depth_multiplier=depth_multiplier,
    )
    return encoder_macs_per_sample(
        seq_len=d1_kept_tokens(mask_strategy),
        embed_dim=int(elastic_model.model.embed_dim),
        active_dim=int(active_heads * head_dim),
        active_ffn_dim=int(active_ffn_dim),
        active_layers=int(active_layers),
    )


def load_dataset(dataset_name: str, root: Path) -> torch.Tensor:
    mat_path = root / dataset_name / "X_test.mat"
    return load_channel_tensor(mat_path, split="test")


def iterate_batches(tensor: torch.Tensor, batch_size: int):
    for start in range(0, int(tensor.shape[0]), int(batch_size)):
        yield tensor[start : start + int(batch_size)]


def extract_outputs(
    model: torch.nn.Module,
    batch: torch.Tensor,
    *,
    dataset_name: str,
    mask_strategy: str,
    mask_ratio: float,
    width_multiplier: float | None = None,
    depth_multiplier: float | None = None,
):
    if width_multiplier is not None and depth_multiplier is not None:
        result = model(
            batch,
            width_multiplier=float(width_multiplier),
            depth_multiplier=float(depth_multiplier),
            return_encoder_state=False,
            mask_ratio=float(mask_ratio),
            mask_strategy=str(mask_strategy),
            data=dataset_name,
        )
        return result.model_output
    return model(
        batch,
        mask_ratio=float(mask_ratio),
        mask_strategy=str(mask_strategy),
        data=dataset_name,
    )


def batch_nmse(outputs) -> tuple[float, int]:
    _, _, pred, target, mask = outputs
    batch_size = int(pred.shape[0])
    pred_mask = pred.squeeze(dim=2)
    target_mask = target.squeeze(dim=2)
    pred_flat = pred_mask[mask == 1].reshape(batch_size, -1)
    target_flat = target_mask[mask == 1].reshape(batch_size, -1)
    sample_nmse = (torch.abs(target_flat - pred_flat) ** 2).mean(dim=1) / (
        torch.abs(target_flat) ** 2
    ).mean(dim=1).clamp_min(1e-8)
    return float(sample_nmse.sum().item()), int(sample_nmse.numel())


def evaluate_nmse(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    dataset_name: str,
    mask_strategy: str,
    batch_size: int,
    device: torch.device,
    width_multiplier: float | None = None,
    depth_multiplier: float | None = None,
) -> float:
    total_nmse = 0.0
    total_count = 0
    with torch.inference_mode():
        for batch in iterate_batches(tensor, batch_size):
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = extract_outputs(
                model,
                batch,
                dataset_name=dataset_name,
                mask_strategy=mask_strategy,
                mask_ratio=0.5,
                width_multiplier=width_multiplier,
                depth_multiplier=depth_multiplier,
            )
            nmse_sum, count = batch_nmse(outputs)
            total_nmse += nmse_sum
            total_count += count
    return float(total_nmse / max(total_count, 1))


class ProfilingWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        dataset_name: str,
        mask_strategy: str,
        width_multiplier: float | None = None,
        depth_multiplier: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataset_name = dataset_name
        self.mask_strategy = mask_strategy
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = extract_outputs(
            self.model,
            x,
            dataset_name=self.dataset_name,
            mask_strategy=self.mask_strategy,
            mask_ratio=0.5,
            width_multiplier=self.width_multiplier,
            depth_multiplier=self.depth_multiplier,
        )
        return outputs[0]


def benchmark_latency(
    wrapper: torch.nn.Module,
    sample: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    wrapper.eval()
    sample = sample.to(device=device, dtype=torch.float32, non_blocking=True)
    with torch.inference_mode():
        if device.type == "cuda":
            for _ in range(warmup):
                _ = wrapper(sample)
            torch.cuda.synchronize(device)
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            timings: list[float] = []
            for _ in range(repeats):
                starter.record()
                _ = wrapper(sample)
                ender.record()
                torch.cuda.synchronize(device)
                timings.append(float(starter.elapsed_time(ender)))
            return float(np.mean(timings))
        start = time.perf_counter()
        for _ in range(repeats):
            _ = wrapper(sample)
        elapsed = (time.perf_counter() - start) * 1000.0
        return float(elapsed / max(repeats, 1))


def measure_macs(
    wrapper: torch.nn.Module,
    sample: torch.Tensor,
) -> tuple[float, float]:
    parameter = next(wrapper.parameters(), None)
    original_device = parameter.device if parameter is not None else torch.device("cpu")
    wrapper_cpu = wrapper.to("cpu").eval()
    sample_cpu = sample.to(dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        macs, params = profile(wrapper_cpu, inputs=(sample_cpu,), verbose=False)
    wrapper.to(original_device)
    return float(macs), float(params)


def format_subnet(width: float, depth: float) -> str:
    return f"w{width:g}_d{depth:g}"


def compute_pareto_front(df: pd.DataFrame, x_key: str) -> pd.DataFrame:
    ordered = df.sort_values([x_key, "nmse"], ascending=[True, True])
    selected_indices: list[int] = []
    best_nmse = float("inf")
    for index, row in ordered.iterrows():
        nmse = float(row["nmse"])
        if nmse < best_nmse - 1e-12:
            selected_indices.append(index)
            best_nmse = nmse
    return ordered.loc[selected_indices]


def plot_tradeoff(
    *,
    subnet_df: pd.DataFrame,
    size_df: pd.DataFrame,
    strujepa_size_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(bottom=0.18, top=0.92, hspace=0.32, wspace=0.24)

    task_meta = {
        "temporal": {"row": 0, "title": "Temporal Prediction"},
        "fre": {"row": 1, "title": "Frequency Prediction"},
    }
    raw_color = "#1f77b4"
    raw_marker = "s"
    size_styles = {
        "base": {"color": "#d65f5f", "marker": "o", "label": "Stru-JEPA Base"},
        "small": {"color": "#ff7f0e", "marker": "^", "label": "Stru-JEPA Small"},
        "little": {"color": "#2ca02c", "marker": "D", "label": "Stru-JEPA Little"},
        "tiny": {"color": "#9467bd", "marker": "P", "label": "Stru-JEPA Tiny"},
    }

    def draw_curve(
        ax: plt.Axes,
        df: pd.DataFrame,
        *,
        x_key: str,
        color: str,
        marker: str,
        label: str,
        annotate_offset: tuple[int, int],
        zorder: int,
        annotate: bool,
    ) -> None:
        if df.empty:
            return
        pareto_df = compute_pareto_front(df, x_key)
        ax.scatter(
            df[x_key],
            df["nmse"],
            s=52,
            marker=marker,
            color=color,
            alpha=0.22,
            linewidths=0.0,
            zorder=max(1, zorder - 1),
        )
        ax.plot(
            pareto_df[x_key],
            pareto_df["nmse"],
            marker=marker,
            linewidth=2.0,
            color=color,
            label=label,
            alpha=0.95,
            zorder=zorder,
        )
        if annotate:
            for _, row_data in pareto_df.iterrows():
                ax.annotate(
                    row_data["label"],
                    (row_data[x_key], row_data["nmse"]),
                    textcoords="offset points",
                    xytext=annotate_offset,
                    fontsize=7,
                )

    for task_key, meta in task_meta.items():
        row = meta["row"]
        task_subnet = subnet_df[subnet_df["task"] == task_key].copy()
        task_size = size_df[
            (size_df["task"] == task_key) & (size_df["label"].isin(PLOT_RAW_MODEL_SIZES))
        ].copy()
        task_size["size_rank"] = task_size["label"].map(
            {name: idx for idx, name in enumerate(RAW_MODEL_SIZES)}
        )
        task_size = task_size.sort_values("size_rank").copy()

        ax_latency = axes[row, 0]
        ax_macs = axes[row, 1]

        draw_curve(
            ax_latency,
            task_size,
            x_key="latency_ms",
            color=raw_color,
            marker=raw_marker,
            label="Original WiFo Weights",
            annotate_offset=(4, -12),
            zorder=4,
            annotate=True,
        )
        draw_curve(
            ax_macs,
            task_size,
            x_key="macs_g",
            color=raw_color,
            marker=raw_marker,
            label="Original WiFo Weights",
            annotate_offset=(4, -12),
            zorder=4,
            annotate=True,
        )
        if strujepa_size_df is not None and not strujepa_size_df.empty:
            for size_name in ("base", "small", "little", "tiny"):
                style = size_styles[size_name]
                size_rows = strujepa_size_df[
                    (strujepa_size_df["task"] == task_key)
                    & (strujepa_size_df["size_label"] == size_name)
                ].copy()
                if size_rows.empty:
                    continue
                draw_curve(
                    ax_latency,
                    size_rows,
                    x_key="latency_ms",
                    color=style["color"],
                    marker=style["marker"],
                    label=style["label"],
                    annotate_offset=(4, 5),
                    zorder=5,
                    annotate=True,
                )
                draw_curve(
                    ax_macs,
                    size_rows,
                    x_key="macs_g",
                    color=style["color"],
                    marker=style["marker"],
                    label=style["label"],
                    annotate_offset=(4, 5),
                    zorder=5,
                    annotate=True,
                )

        ax_latency.set_title(f"({chr(97 + row * 2)}) {meta['title']}")
        ax_macs.set_title(f"({chr(98 + row * 2)}) {meta['title']}")
        ax_latency.set_xlabel("Inference Latency (ms)")
        ax_macs.set_xlabel("MACs (G)")
        ax_latency.set_ylabel("NMSE (Lower is Better)")
        ax_macs.set_ylabel("NMSE (Lower is Better)")
        ax_latency.grid(True, alpha=0.25)
        ax_macs.grid(True, alpha=0.25)

    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    deduped: dict[str, object] = {}
    for handle, label in zip(handles, labels_list):
        if label not in deduped:
            deduped[label] = handle
    fig.legend(
        deduped.values(),
        deduped.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(4, max(1, len(deduped))),
        frameon=False,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_markdown_table(df: pd.DataFrame, *, value_column: str, title: str) -> str:
    ordered = df.copy()
    ordered["Dataset"] = pd.Categorical(
        ordered["Dataset"],
        categories=list(FULL_DATASETS) + ["Average"],
        ordered=True,
    )
    ordered = ordered.sort_values("Dataset")
    lines = [f"## {title}", "", "| Dataset | Stru-JEPA NMSE |", "| --- | ---: |"]
    for _, row in ordered.iterrows():
        lines.append(f"| {row['Dataset']} | {row[value_column]:.3f} |")
    return "\n".join(lines)


def parse_dataset_names(dataset_spec: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in str(dataset_spec).split("*") if name.strip())


def infer_model_label(checkpoint_args: dict[str, object]) -> str:
    size = str(checkpoint_args.get("size", "")).strip()
    if size:
        return f"Stru-JEPA-{size.capitalize()}"
    return "Stru-JEPA"


def infer_train_batches_per_epoch(train_log: Path) -> int | None:
    if not train_log.exists():
        return None
    for line in train_log.read_text(encoding="utf-8").splitlines():
        if '"batches_per_rank"' not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        batches = payload.get("batches_per_rank")
        if batches is not None:
            return int(batches)
    return None


def summarize_training_speed(
    run_dir: Path,
    *,
    checkpoint: dict[str, object],
) -> dict[str, float | str]:
    train_log = run_dir / "train.log"
    match = re.search(r"(\d{8}_\d{6})$", run_dir.name)
    if match is not None:
        parsed = time.strptime(match.group(1), "%Y%m%d_%H%M%S")
        start = time.mktime(parsed)
    else:
        start = train_log.stat().st_ctime
    end = train_log.stat().st_mtime
    elapsed_s = float(end - start)
    checkpoint_args = dict(checkpoint.get("args", {}))
    history = list(checkpoint.get("history", []))
    epochs = float(len(history) or int(checkpoint_args.get("epochs", 1) or 1))
    dataset_count = float(max(1, len(parse_dataset_names(str(checkpoint_args.get("dataset", ""))))))
    batch_size = int(checkpoint_args.get("batch_size", 1) or 1)
    validate_every = int(checkpoint_args.get("validate_every", 1) or 1)
    validation_epochs = max(1.0, math.ceil(epochs / max(validate_every, 1)))
    train_samples_per_epoch = dataset_count * 9000.0
    val_samples_per_epoch = dataset_count * 2000.0
    total_samples_per_epoch = train_samples_per_epoch + val_samples_per_epoch
    train_batches_per_epoch = infer_train_batches_per_epoch(train_log)
    if train_batches_per_epoch is None:
        train_batches_per_epoch = math.ceil(9000 / max(batch_size, 1)) * int(dataset_count)
    val_batches_per_epoch = math.ceil(2000 / max(batch_size, 1)) * int(dataset_count)
    total_seen_samples = (train_samples_per_epoch * epochs) + (val_samples_per_epoch * validation_epochs)
    return {
        "elapsed_hours": elapsed_s / 3600.0,
        "elapsed_seconds": elapsed_s,
        "seconds_per_epoch": elapsed_s / epochs,
        "minutes_per_epoch": elapsed_s / epochs / 60.0,
        "train_samples_per_second_est": (train_samples_per_epoch * epochs) / elapsed_s,
        "all_samples_per_second_est": total_seen_samples / elapsed_s,
        "train_batches_per_epoch": train_batches_per_epoch,
        "val_batches_per_epoch": val_batches_per_epoch,
        "epochs": epochs,
        "validation_epochs": validation_epochs,
        "dataset_count": dataset_count,
        "batch_size": batch_size,
        "samples_per_epoch_total_nominal": total_samples_per_epoch,
        "run_dir": str(run_dir),
    }


def build_paper_table_markdown(
    *,
    title: str,
    baselines: dict[str, dict[str, float]],
    strujepa_scores: dict[str, float],
    model_label: str,
) -> str:
    baseline_columns = list(next(iter(baselines.values())).keys())
    header = ["Dataset", *baseline_columns, model_label]
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for dataset_name in list(FULL_DATASETS) + ["Average"]:
        baseline_row = baselines[dataset_name]
        row_values = [dataset_name]
        row_values.extend(f"{baseline_row[column]:.3f}" for column in baseline_columns)
        row_values.append(f"{strujepa_scores[dataset_name]:.3f}")
        lines.append("| " + " | ".join(row_values) + " |")
    return "\n".join(lines)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RUN_DEFAULT / "strujepa_wifo_last.pt",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DEFAULT,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_DIR,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--latency-batch-size", type=int, default=8)
    parser.add_argument("--latency-warmup", type=int, default=15)
    parser.add_argument("--latency-repeats", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / f"analysis_tradeoff_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--size-checkpoint",
        action="append",
        type=Path,
        default=[],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = dict(checkpoint["args"])
    trained_widths = parse_multiplier_string(checkpoint_args["width_multipliers"])
    trained_depths = parse_multiplier_string(checkpoint_args["depth_multipliers"])
    model_label = infer_model_label(checkpoint_args)

    speed_summary = summarize_training_speed(args.run_dir, checkpoint=checkpoint)
    (args.output_dir / "training_speed.json").write_text(
        json.dumps(speed_summary, indent=2),
        encoding="utf-8",
    )

    print("loading cached plot datasets...", flush=True)
    plot_tensors = {name: load_dataset(name, args.dataset_root) for name in PLOT_DATASETS}
    latency_sample = plot_tensors["D1"][: args.latency_batch_size]
    raw_records: list[dict[str, object]] = []
    subnet_records: list[dict[str, object]] = []
    strujepa_size_records: list[dict[str, object]] = []
    table_records: list[dict[str, object]] = []

    print("evaluating raw WiFo sizes...")
    for size in RAW_MODEL_SIZES:
        model = load_raw_model(size, device)
        for task_key, _ in TASKS:
            dataset_scores = []
            for dataset_name in PLOT_DATASETS:
                print(f"  raw size={size} task={task_key} dataset={dataset_name}", flush=True)
                tensor = plot_tensors[dataset_name]
                nmse = evaluate_nmse(
                    model,
                    tensor,
                    dataset_name=dataset_name,
                    mask_strategy=task_key,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                dataset_scores.append(nmse)
            wrapper = ProfilingWrapper(
                model,
                dataset_name="D1",
                mask_strategy=task_key,
            )
            params = float(sum(parameter.numel() for parameter in model.parameters()))
            latency_ms = benchmark_latency(
                wrapper,
                latency_sample,
                device=device,
                warmup=args.latency_warmup,
                repeats=args.latency_repeats,
            )
            raw_records.append(
                {
                    "family": "raw_size",
                    "label": size,
                    "task": task_key,
                    "nmse": float(np.mean(dataset_scores)),
                    "latency_ms": latency_ms,
                    "macs_g": estimate_raw_encoder_macs(model, mask_strategy=task_key) / 1e9,
                    "params_m": params / 1e6,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elastic_model = load_strujepa_model(
        args.checkpoint,
        device,
        width_multipliers=trained_widths,
        depth_multipliers=trained_depths,
    )

    print("evaluating Stru-JEPA full model on D1-D16 tables...")
    for task_key, _ in TASKS:
        scores = []
        for dataset_name in FULL_DATASETS:
            print(f"  table task={task_key} dataset={dataset_name}", flush=True)
            tensor = plot_tensors.get(dataset_name)
            if tensor is None:
                tensor = load_dataset(dataset_name, args.dataset_root)
            nmse = evaluate_nmse(
                elastic_model,
                tensor,
                dataset_name=dataset_name,
                mask_strategy=task_key,
                batch_size=args.eval_batch_size,
                device=device,
                width_multiplier=1.0,
                depth_multiplier=1.0,
            )
            scores.append(nmse)
            table_records.append(
                {
                    "task": task_key,
                    "Dataset": dataset_name,
                    "StruJEPA_NMSE": nmse,
                }
            )
        table_records.append(
            {
                "task": task_key,
                "Dataset": "Average",
                "StruJEPA_NMSE": float(np.mean(scores)),
            }
        )

    print("evaluating Stru-JEPA subnets...")
    for depth in trained_depths or DEFAULT_SUBNET_DEPTHS:
        for width in trained_widths or DEFAULT_SUBNET_WIDTHS:
            label = format_subnet(width, depth)
            for task_key, _ in TASKS:
                dataset_scores = []
                for dataset_name in PLOT_DATASETS:
                    print(
                        f"  subnet={label} task={task_key} dataset={dataset_name}",
                        flush=True,
                    )
                    tensor = plot_tensors[dataset_name]
                    nmse = evaluate_nmse(
                        elastic_model,
                        tensor,
                        dataset_name=dataset_name,
                        mask_strategy=task_key,
                        batch_size=args.eval_batch_size,
                        device=device,
                        width_multiplier=width,
                        depth_multiplier=depth,
                    )
                    dataset_scores.append(nmse)
                wrapper = ProfilingWrapper(
                    elastic_model,
                    dataset_name="D1",
                    mask_strategy=task_key,
                    width_multiplier=width,
                    depth_multiplier=depth,
                )
                params = float(sum(parameter.numel() for parameter in elastic_model.model.parameters()))
                latency_ms = benchmark_latency(
                    wrapper,
                    latency_sample,
                    device=device,
                    warmup=args.latency_warmup,
                    repeats=args.latency_repeats,
                )
                subnet_records.append(
                    {
                        "family": "strujepa_subnet",
                        "label": label,
                        "task": task_key,
                        "width_multiplier": width,
                        "depth_multiplier": depth,
                        "nmse": float(np.mean(dataset_scores)),
                        "latency_ms": latency_ms,
                        "macs_g": estimate_elastic_encoder_macs(
                            elastic_model,
                            mask_strategy=task_key,
                            width_multiplier=width,
                            depth_multiplier=depth,
                        )
                        / 1e9,
                        "params_m": params / 1e6,
                    }
                )

    print("evaluating Stru-JEPA size models...")
    size_checkpoints = dedupe_paths([args.checkpoint, *list(args.size_checkpoint)])
    for checkpoint_path in size_checkpoints:
        size_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        size_args = dict(size_checkpoint["args"])
        size_label = str(size_args["size"])
        size_widths = parse_multiplier_string(size_args["width_multipliers"])
        size_depths = parse_multiplier_string(size_args["depth_multipliers"])
        size_model = load_strujepa_model(
            checkpoint_path,
            device,
            width_multipliers=size_widths,
            depth_multipliers=size_depths,
        )
        params = float(sum(parameter.numel() for parameter in size_model.model.parameters()))
        for depth in size_depths or DEFAULT_SUBNET_DEPTHS:
            for width in size_widths or DEFAULT_SUBNET_WIDTHS:
                subnet_label = format_subnet(width, depth)
                for task_key, _ in TASKS:
                    dataset_scores = []
                    for dataset_name in PLOT_DATASETS:
                        print(
                            f"  strujepa_size={size_label} subnet={subnet_label} task={task_key} dataset={dataset_name}",
                            flush=True,
                        )
                        tensor = plot_tensors[dataset_name]
                        nmse = evaluate_nmse(
                            size_model,
                            tensor,
                            dataset_name=dataset_name,
                            mask_strategy=task_key,
                            batch_size=args.eval_batch_size,
                            device=device,
                            width_multiplier=width,
                            depth_multiplier=depth,
                        )
                        dataset_scores.append(nmse)
                    wrapper = ProfilingWrapper(
                        size_model,
                        dataset_name="D1",
                        mask_strategy=task_key,
                        width_multiplier=width,
                        depth_multiplier=depth,
                    )
                    latency_ms = benchmark_latency(
                        wrapper,
                        latency_sample,
                        device=device,
                        warmup=args.latency_warmup,
                        repeats=args.latency_repeats,
                    )
                    strujepa_size_records.append(
                        {
                            "family": "strujepa_size",
                            "size_label": size_label,
                            "label": subnet_label,
                            "task": task_key,
                            "width_multiplier": width,
                            "depth_multiplier": depth,
                            "nmse": float(np.mean(dataset_scores)),
                            "latency_ms": latency_ms,
                            "macs_g": estimate_elastic_encoder_macs(
                                size_model,
                                mask_strategy=task_key,
                                width_multiplier=width,
                                depth_multiplier=depth,
                            )
                            / 1e9,
                            "params_m": params / 1e6,
                            "checkpoint": str(checkpoint_path),
                        }
                    )
        del size_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_df = pd.DataFrame(raw_records)
    subnet_df = pd.DataFrame(subnet_records)
    strujepa_size_df = pd.DataFrame(strujepa_size_records)
    table_df = pd.DataFrame(table_records)

    raw_df.to_csv(args.output_dir / "raw_size_tradeoff.csv", index=False)
    subnet_df.to_csv(args.output_dir / "strujepa_subnet_tradeoff.csv", index=False)
    strujepa_size_df.to_csv(args.output_dir / "strujepa_size_tradeoff.csv", index=False)
    table_df.to_csv(args.output_dir / "strujepa_table_results.csv", index=False)

    plot_tradeoff(
        subnet_df=subnet_df,
        size_df=raw_df,
        strujepa_size_df=strujepa_size_df,
        output_path=args.output_dir / "tradeoff_2x2.png",
    )

    temporal_table = build_markdown_table(
        table_df[table_df["task"] == "temporal"],
        value_column="StruJEPA_NMSE",
        title="Stru-JEPA Temporal Prediction (D1-D16)",
    )
    frequency_table = build_markdown_table(
        table_df[table_df["task"] == "fre"],
        value_column="StruJEPA_NMSE",
        title="Stru-JEPA Frequency Prediction (D1-D16)",
    )
    summary_lines = [
        "# Trade-off Analysis",
        "",
        "Reference cost setting: D1 test samples, batch size 8.",
        "Reference compute setting: D1 encoder MACs per sample.",
        "Reference performance setting: mean NMSE over D1, D5, D9, D15.",
        "",
        "## Training Speed",
        "",
        f"- total_time_hours: {speed_summary['elapsed_hours']:.3f}",
        f"- minutes_per_epoch: {speed_summary['minutes_per_epoch']:.2f}",
        f"- train_samples_per_second_est: {speed_summary['train_samples_per_second_est']:.2f}",
        f"- all_samples_per_second_est: {speed_summary['all_samples_per_second_est']:.2f}",
        f"- model_label: {model_label}",
        f"- subnet_sampling_mode: {checkpoint_args.get('subnet_sampling_mode', 'unknown')}",
        f"- raw_sizes_plotted: {', '.join(PLOT_RAW_MODEL_SIZES)}",
        f"- widths: {', '.join(f'{value:g}' for value in trained_widths)}",
        f"- depths: {', '.join(f'{value:g}' for value in trained_depths)}",
        "",
        temporal_table,
        "",
        frequency_table,
        "",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    temporal_scores = {
        row["Dataset"]: float(row["StruJEPA_NMSE"])
        for _, row in table_df[table_df["task"] == "temporal"].iterrows()
    }
    frequency_scores = {
        row["Dataset"]: float(row["StruJEPA_NMSE"])
        for _, row in table_df[table_df["task"] == "fre"].iterrows()
    }
    paper_lines = [
        f"Checkpoint: `{args.checkpoint}`",
        "",
        f"Subnet sampling: `{checkpoint_args.get('subnet_sampling_mode', 'unknown')}`",
        "",
        build_paper_table_markdown(
            title="Time-domain",
            baselines=PAPER_TIME_BASELINES,
            strujepa_scores=temporal_scores,
            model_label=model_label,
        ),
        "",
        build_paper_table_markdown(
            title="Frequency-domain",
            baselines=PAPER_FREQUENCY_BASELINES,
            strujepa_scores=frequency_scores,
            model_label=model_label,
        ),
        "",
    ]
    (args.output_dir / "paper_plus_strujepa_tables.md").write_text(
        "\n".join(paper_lines),
        encoding="utf-8",
    )
    print(f"analysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()

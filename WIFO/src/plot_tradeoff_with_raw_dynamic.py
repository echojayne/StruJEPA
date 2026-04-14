from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from analyze_tradeoff import (
    DATASET_DIR,
    PLOT_DATASETS,
    ProfilingWrapper,
    benchmark_latency,
    compute_pareto_front,
    estimate_elastic_encoder_macs,
    evaluate_nmse,
    load_dataset,
    load_raw_model,
)
from elastic_wifo import elasticize_wifo


PALETTE = {
    "base": "#81021F",
    "small": "#15559A",
    "little": "#024943",
    "tiny": "#EF9D1E",
}
BG = "#F9F6E5"
INK = "#2B313F"
GRID = "#B7C8D6"
SIZE_ORDER = ("base", "small", "little", "tiny")
TASK_META = {
    "temporal": {"row": 0, "title": "Temporal Prediction"},
    "fre": {"row": 1, "title": "Frequency Prediction"},
}
LINE_META = {
    "strujepa": {"linestyle": "-", "marker": "o", "filled": True, "label": "Stru-JEPA"},
    "raw_dynamic": {"linestyle": "--", "marker": "o", "filled": False, "label": "Raw WiFo Dynamicized"},
}


def beautify_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def load_subnet_specs(strujepa_df: pd.DataFrame) -> list[tuple[float, float]]:
    specs = (
        strujepa_df[["width_multiplier", "depth_multiplier"]]
        .drop_duplicates()
        .sort_values(["depth_multiplier", "width_multiplier"], ascending=[False, False])
    )
    return [
        (float(row["width_multiplier"]), float(row["depth_multiplier"]))
        for _, row in specs.iterrows()
    ]


def evaluate_raw_dynamic_tradeoff(
    *,
    dataset_root: Path,
    device: torch.device,
    eval_batch_size: int,
    latency_batch_size: int,
    latency_warmup: int,
    latency_repeats: int,
    subnet_specs: list[tuple[float, float]],
) -> pd.DataFrame:
    plot_tensors = {name: load_dataset(name, dataset_root) for name in PLOT_DATASETS}
    latency_sample = plot_tensors["D1"][:latency_batch_size]
    rows: list[dict[str, object]] = []

    for size_name in SIZE_ORDER[::-1]:
        print(f"evaluating raw dynamic size={size_name}", flush=True)
        base_model = load_raw_model(size_name, device)
        elastic_model = elasticize_wifo(
            base_model,
            width_multipliers=tuple(sorted({spec[0] for spec in subnet_specs}, reverse=True)),
            depth_multipliers=tuple(sorted({spec[1] for spec in subnet_specs}, reverse=True)),
            copy_model=True,
        )
        elastic_model.to(device)
        elastic_model.eval()
        params = float(sum(parameter.numel() for parameter in elastic_model.model.parameters()))

        for width, depth in subnet_specs:
            label = f"w{width:g}_d{depth:g}"
            for task_key in ("temporal", "fre"):
                dataset_scores = []
                for dataset_name in PLOT_DATASETS:
                    print(
                        f"  raw_dynamic size={size_name} subnet={label} task={task_key} dataset={dataset_name}",
                        flush=True,
                    )
                    nmse = evaluate_nmse(
                        elastic_model,
                        plot_tensors[dataset_name],
                        dataset_name=dataset_name,
                        mask_strategy=task_key,
                        batch_size=eval_batch_size,
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
                latency_ms = benchmark_latency(
                    wrapper,
                    latency_sample,
                    device=device,
                    warmup=latency_warmup,
                    repeats=latency_repeats,
                )
                rows.append(
                    {
                        "family": "raw_dynamic_size",
                        "size_label": size_name,
                        "label": label,
                        "task": task_key,
                        "width_multiplier": width,
                        "depth_multiplier": depth,
                        "nmse": float(sum(dataset_scores) / max(len(dataset_scores), 1)),
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

        del elastic_model
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def draw_curve(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_key: str,
    color: str,
    linestyle: str,
    marker: str,
    filled: bool,
    alpha: float,
    zorder: int,
) -> None:
    if df.empty:
        return
    pareto_df = compute_pareto_front(df, x_key)
    ax.plot(
        pareto_df[x_key],
        pareto_df["nmse"],
        color=color,
        linestyle=linestyle,
        linewidth=2.2,
        marker=marker,
        markersize=5.2,
        markerfacecolor=color if filled else BG,
        markeredgecolor=color,
        markeredgewidth=1.3,
        alpha=alpha,
        zorder=zorder,
    )


def plot_overlay(
    *,
    strujepa_df: pd.DataFrame,
    raw_dynamic_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    fig.set_facecolor(BG)
    fig.subplots_adjust(bottom=0.16, top=0.92, hspace=0.34, wspace=0.24)

    for task_key, meta in TASK_META.items():
        ax_latency = axes[meta["row"], 0]
        ax_macs = axes[meta["row"], 1]
        beautify_axes(ax_latency)
        beautify_axes(ax_macs)

        for size_name in SIZE_ORDER:
            color = PALETTE[size_name]
            strujepa_rows = strujepa_df[
                (strujepa_df["task"] == task_key) & (strujepa_df["size_label"] == size_name)
            ].copy()
            raw_dynamic_rows = raw_dynamic_df[
                (raw_dynamic_df["task"] == task_key) & (raw_dynamic_df["size_label"] == size_name)
            ].copy()

            draw_curve(
                ax_latency,
                strujepa_rows,
                x_key="latency_ms",
                color=color,
                linestyle=LINE_META["strujepa"]["linestyle"],
                marker=LINE_META["strujepa"]["marker"],
                filled=LINE_META["strujepa"]["filled"],
                alpha=0.96,
                zorder=4,
            )
            draw_curve(
                ax_macs,
                strujepa_rows,
                x_key="macs_g",
                color=color,
                linestyle=LINE_META["strujepa"]["linestyle"],
                marker=LINE_META["strujepa"]["marker"],
                filled=LINE_META["strujepa"]["filled"],
                alpha=0.96,
                zorder=4,
            )
            draw_curve(
                ax_latency,
                raw_dynamic_rows,
                x_key="latency_ms",
                color=color,
                linestyle=LINE_META["raw_dynamic"]["linestyle"],
                marker=LINE_META["raw_dynamic"]["marker"],
                filled=LINE_META["raw_dynamic"]["filled"],
                alpha=0.95,
                zorder=3,
            )
            draw_curve(
                ax_macs,
                raw_dynamic_rows,
                x_key="macs_g",
                color=color,
                linestyle=LINE_META["raw_dynamic"]["linestyle"],
                marker=LINE_META["raw_dynamic"]["marker"],
                filled=LINE_META["raw_dynamic"]["filled"],
                alpha=0.95,
                zorder=3,
            )

        panel_base = ord("a") + meta["row"] * 2
        ax_latency.set_title(f"({chr(panel_base)}) {meta['title']}")
        ax_macs.set_title(f"({chr(panel_base + 1)}) {meta['title']}")
        ax_latency.set_xlabel("Inference Latency (ms)")
        ax_macs.set_xlabel("MACs (G)")
        ax_latency.set_ylabel("NMSE (Lower is Better)")
        ax_macs.set_ylabel("NMSE (Lower is Better)")

    size_handles = [
        Line2D([0], [0], color=PALETTE[size_name], lw=2.6, label=size_name.capitalize())
        for size_name in SIZE_ORDER
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            lw=2.4,
            linestyle=meta["linestyle"],
            marker=meta["marker"],
            markersize=5.2,
            markerfacecolor=INK if meta["filled"] else BG,
            markeredgecolor=INK,
            markeredgewidth=1.2,
            label=meta["label"],
        )
        for meta in LINE_META.values()
    ]

    legend_handles = [*size_handles, *method_handles]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        frameon=False,
        fontsize=10,
        handlelength=2.6,
        columnspacing=1.4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--latency-batch-size", type=int, default=8)
    parser.add_argument("--latency-warmup", type=int, default=25)
    parser.add_argument("--latency-repeats", type=int, default=80)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    strujepa_df = pd.read_csv(analysis_dir / "strujepa_size_tradeoff.csv")
    subnet_specs = load_subnet_specs(strujepa_df)
    raw_dynamic_csv = analysis_dir / "raw_dynamic_size_tradeoff.csv"
    if args.reuse_existing and raw_dynamic_csv.exists():
        raw_dynamic_df = pd.read_csv(raw_dynamic_csv)
    else:
        raw_dynamic_df = evaluate_raw_dynamic_tradeoff(
            dataset_root=args.dataset_root.resolve(),
            device=torch.device(args.device),
            eval_batch_size=args.eval_batch_size,
            latency_batch_size=args.latency_batch_size,
            latency_warmup=args.latency_warmup,
            latency_repeats=args.latency_repeats,
            subnet_specs=subnet_specs,
        )
        raw_dynamic_df.to_csv(raw_dynamic_csv, index=False)
    plot_overlay(
        strujepa_df=strujepa_df,
        raw_dynamic_df=raw_dynamic_df,
        output_path=analysis_dir / "tradeoff_2x2.png",
    )
    plot_overlay(
        strujepa_df=strujepa_df,
        raw_dynamic_df=raw_dynamic_df,
        output_path=analysis_dir / "tradeoff_2x2_raw_dynamic.png",
    )
    print(analysis_dir / "tradeoff_2x2.png")


if __name__ == "__main__":
    main()

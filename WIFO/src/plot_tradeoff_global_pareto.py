from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


RAW_ORDER = ("tiny", "little", "small", "base")
SIZE_ABBR = {"tiny": "T", "little": "L", "small": "S", "base": "B"}
SUBNET_ABBR = {
    "w1_d1": "1,1",
    "w1_d0.5": "1,1/2",
    "w1_d0.166667": "1,1/6",
    "w0.5_d1": "1/2,1",
    "w0.5_d0.5": "1/2,1/2",
    "w0.5_d0.166667": "1/2,1/6",
    "w0.125_d1": "1/8,1",
    "w0.125_d0.5": "1/8,1/2",
    "w0.125_d0.166667": "1/8,1/6",
}
PALETTE = {
    "j_red": "#81021F",
    "j_blue": "#003153",
    "ivory": "#F9F6E5",
    "ink": "#312520",
    "grid": "#B7C8D6",
}

ANNOTATE_OFFSETS = {
    "tiny": (6, 8),
    "little": (6, -12),
    "small": (6, 8),
    "base": (6, -12),
}


def compute_pareto_front(df: pd.DataFrame, x_key: str) -> pd.DataFrame:
    ordered = df.sort_values([x_key, "nmse", "label"]).reset_index(drop=True)
    kept_rows: list[int] = []
    best_nmse = float("inf")
    for idx, row in ordered.iterrows():
        nmse = float(row["nmse"])
        if nmse < best_nmse - 1e-12:
            kept_rows.append(idx)
            best_nmse = nmse
    return ordered.iloc[kept_rows].copy()


def beautify_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PALETTE["ivory"])
    ax.grid(True, color=PALETTE["grid"], alpha=0.35, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    ax.tick_params(colors=PALETTE["ink"], labelsize=10)
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.title.set_color(PALETTE["ink"])


def annotate_raw_sizes(ax: plt.Axes, df: pd.DataFrame, *, x_key: str) -> None:
    for _, row in df.iterrows():
        size_name = str(row["label"])
        xytext = ANNOTATE_OFFSETS.get(size_name, (6, 6))
        ax.annotate(
            size_name,
            (float(row[x_key]), float(row["nmse"])),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
            color=PALETTE["j_blue"],
            alpha=0.95,
            fontweight="medium",
        )


def annotate_strujepa_sizes(ax: plt.Axes, df: pd.DataFrame, *, x_key: str) -> None:
    seen: set[str] = set()
    for _, row in df.iterrows():
        size_name = str(row["size_label"])
        if size_name in seen:
            continue
        seen.add(size_name)
        xytext = ANNOTATE_OFFSETS.get(size_name, (6, 6))
        ax.annotate(
            size_name,
            (float(row[x_key]), float(row["nmse"])),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
            color=PALETTE["j_red"],
            alpha=0.95,
            fontweight="medium",
        )


def plot_global_pareto(*, raw_df: pd.DataFrame, strujepa_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "medium",
            "figure.facecolor": PALETTE["ivory"],
            "savefig.facecolor": PALETTE["ivory"],
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    fig.set_dpi(600)
    fig.set_facecolor(PALETTE["ivory"])
    fig.subplots_adjust(bottom=0.14, top=0.92, hspace=0.34, wspace=0.24)

    task_meta = {
        "temporal": {"row": 0, "title": "Temporal Prediction"},
        "fre": {"row": 1, "title": "Frequency Prediction"},
    }

    for task_key, meta in task_meta.items():
        raw_task = raw_df[(raw_df["task"] == task_key) & (raw_df["label"].isin(RAW_ORDER))].copy()
        raw_task["size_rank"] = raw_task["label"].map({name: idx for idx, name in enumerate(RAW_ORDER)})
        raw_task = raw_task.sort_values("size_rank").copy()
        raw_task_front_latency = compute_pareto_front(raw_task, "latency_ms")
        raw_task_front_macs = compute_pareto_front(raw_task, "macs_g")

        strujepa_task = strujepa_df[strujepa_df["task"] == task_key].copy()
        strujepa_front_latency = compute_pareto_front(strujepa_task, "latency_ms")
        strujepa_front_macs = compute_pareto_front(strujepa_task, "macs_g")

        ax_latency = axes[meta["row"], 0]
        ax_macs = axes[meta["row"], 1]
        for ax in (ax_latency, ax_macs):
            beautify_axes(ax)

        ax_latency.scatter(
            raw_task["latency_ms"],
            raw_task["nmse"],
            s=52,
            color=PALETTE["j_blue"],
            alpha=0.18,
            linewidths=0.0,
            zorder=1,
        )
        ax_macs.scatter(
            raw_task["macs_g"],
            raw_task["nmse"],
            s=52,
            color=PALETTE["j_blue"],
            alpha=0.18,
            linewidths=0.0,
            zorder=1,
        )
        ax_latency.plot(
            raw_task_front_latency["latency_ms"],
            raw_task_front_latency["nmse"],
            color=PALETTE["j_blue"],
            linewidth=2.2,
            marker="s",
            markersize=5.5,
            label="Original WiFo Weights",
            zorder=3,
        )
        ax_macs.plot(
            raw_task_front_macs["macs_g"],
            raw_task_front_macs["nmse"],
            color=PALETTE["j_blue"],
            linewidth=2.2,
            marker="s",
            markersize=5.5,
            label="Original WiFo Weights",
            zorder=3,
        )
        annotate_raw_sizes(ax_latency, raw_task_front_latency, x_key="latency_ms")
        annotate_raw_sizes(ax_macs, raw_task_front_macs, x_key="macs_g")

        ax_latency.scatter(
            strujepa_task["latency_ms"],
            strujepa_task["nmse"],
            s=30,
            color=PALETTE["j_red"],
            alpha=0.10,
            linewidths=0.0,
            zorder=1,
        )
        ax_macs.scatter(
            strujepa_task["macs_g"],
            strujepa_task["nmse"],
            s=30,
            color=PALETTE["j_red"],
            alpha=0.10,
            linewidths=0.0,
            zorder=1,
        )
        ax_latency.plot(
            strujepa_front_latency["latency_ms"],
            strujepa_front_latency["nmse"],
            color=PALETTE["j_red"],
            linewidth=2.6,
            marker="o",
            markersize=5.2,
            label="Stru-JEPA Global Pareto Front",
            zorder=4,
        )
        ax_macs.plot(
            strujepa_front_macs["macs_g"],
            strujepa_front_macs["nmse"],
            color=PALETTE["j_red"],
            linewidth=2.6,
            marker="o",
            markersize=5.2,
            label="Stru-JEPA Global Pareto Front",
            zorder=4,
        )
        annotate_strujepa_sizes(ax_latency, strujepa_front_latency, x_key="latency_ms")
        annotate_strujepa_sizes(ax_macs, strujepa_front_macs, x_key="macs_g")

        panel_base = ord("a") + meta["row"] * 2
        ax_latency.set_title(f"({chr(panel_base)}) {meta['title']}")
        ax_macs.set_title(f"({chr(panel_base + 1)}) {meta['title']}")
        ax_latency.set_xlabel("Inference Latency (ms)")
        ax_macs.set_xlabel("MACs (G)")
        ax_latency.set_ylabel("NMSE (Lower is Better)")
        ax_macs.set_ylabel("NMSE (Lower is Better)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    deduped: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in deduped:
            deduped[label] = handle
    fig.legend(
        deduped.values(),
        deduped.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="tradeoff_2x2_global_pareto.png")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    raw_df = pd.read_csv(analysis_dir / "raw_size_tradeoff.csv")
    strujepa_df = pd.read_csv(analysis_dir / "strujepa_size_tradeoff.csv")
    plot_global_pareto(
        raw_df=raw_df,
        strujepa_df=strujepa_df,
        output_path=analysis_dir / args.output_name,
    )
    print(analysis_dir / args.output_name)


if __name__ == "__main__":
    main()

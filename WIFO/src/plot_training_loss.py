# coding=utf-8
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_progress_rows(train_log: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with train_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "train_progress":
                continue
            epoch = int(payload["epoch"])
            batch = int(payload["batch"])
            batches_per_rank = int(payload["batches_per_rank"])
            global_batch = (epoch - 1) * batches_per_rank + batch
            rows.append(
                {
                    "epoch": float(epoch),
                    "batch": float(batch),
                    "batches_per_rank": float(batches_per_rank),
                    "global_batch": float(global_batch),
                    "loss": float(payload["loss"]),
                    "avg_loss": float(payload.get("avg_loss", payload["loss"])),
                    "full_nmse": float(payload.get("full_nmse", float("nan"))),
                    "avg_full_nmse": float(payload.get("avg_full_nmse", payload.get("full_nmse", float("nan")))),
                }
            )
    return rows


def load_history(history_path: Path) -> list[dict[str, float]]:
    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [dict(item) for item in payload]


def offset_progress_rows(
    rows: list[dict[str, float]],
    epoch_offset: int,
    batch_offset: int,
) -> list[dict[str, float]]:
    adjusted: list[dict[str, float]] = []
    for row in rows:
        item = dict(row)
        item["epoch"] = float(int(row["epoch"]) + epoch_offset)
        item["global_batch"] = float(row["global_batch"] + batch_offset)
        adjusted.append(item)
    return adjusted


def derive_epoch_rows(progress_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    per_epoch: dict[int, dict[str, float]] = {}
    for row in progress_rows:
        epoch = int(row["epoch"])
        current = per_epoch.get(epoch)
        if current is None or int(row["batch"]) > int(current["batch"]):
            per_epoch[epoch] = dict(row)

    derived: list[dict[str, float]] = []
    for epoch in sorted(per_epoch):
        row = per_epoch[epoch]
        derived.append(
            {
                "epoch": float(epoch),
                "train_loss": float(row["avg_loss"]),
                "train_nmse": float(row["avg_full_nmse"]),
                "complete": 1.0 if int(row["batch"]) == int(row["batches_per_rank"]) else 0.0,
            }
        )
    return derived


def merge_run_series(run_dirs: list[Path]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    combined_progress: list[dict[str, float]] = []
    combined_epochs: list[dict[str, float]] = []
    epoch_offset = 0
    batch_offset = 0

    for run_dir in run_dirs:
        progress_rows = load_progress_rows(run_dir / "train.log")
        if not progress_rows:
            continue

        adjusted_progress = offset_progress_rows(progress_rows, epoch_offset, batch_offset)
        combined_progress.extend(adjusted_progress)

        history_path = run_dir / "history.json"
        if history_path.exists():
            history_rows = load_history(history_path)
            for row in history_rows:
                item = dict(row)
                item["epoch"] = float(int(item["epoch"]) + epoch_offset)
                item["complete"] = 1.0
                combined_epochs.append(item)
            completed_epochs = max(int(item["epoch"]) for item in history_rows)
        else:
            derived_rows = derive_epoch_rows(adjusted_progress)
            combined_epochs.extend(derived_rows)
            completed_epochs = 0
            for item in derived_rows:
                if bool(item["complete"]):
                    completed_epochs = max(completed_epochs, int(item["epoch"]) - epoch_offset)

        batch_offset = int(max(item["global_batch"] for item in adjusted_progress))
        epoch_offset += completed_epochs

    return combined_progress, combined_epochs


def split_complete(rows: list[dict[str, float]], key: str) -> tuple[list[float], list[float], list[float], list[float]]:
    complete_x: list[float] = []
    complete_y: list[float] = []
    partial_x: list[float] = []
    partial_y: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            y = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(y):
            continue
        if bool(row.get("complete", 1.0)):
            complete_x.append(float(row["epoch"]))
            complete_y.append(y)
        else:
            partial_x.append(float(row["epoch"]))
            partial_y.append(y)
    return complete_x, complete_y, partial_x, partial_y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    progress_rows, epoch_rows = merge_run_series(args.run_dir)
    if not progress_rows:
        raise RuntimeError(f"no train_progress rows found in {[str(item) for item in args.run_dir]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    if args.title:
        fig.suptitle(args.title, fontsize=14)

    ax = axes[0]
    ax.plot(
        [row["global_batch"] for row in progress_rows],
        [row["avg_loss"] for row in progress_rows],
        color="#d65f5f",
        linewidth=1.8,
    )
    epoch_boundaries = sorted(
        {
            int(row["global_batch"])
            for row in progress_rows
            if int(row["batch"]) == int(row["batches_per_rank"])
        }
    )
    for boundary in epoch_boundaries[:-1]:
        ax.axvline(boundary, color="#999999", linewidth=0.8, alpha=0.35)
    ax.set_title("Batch-Level Running Average Loss")
    ax.set_xlabel("Global Batch")
    ax.set_ylabel("Avg Loss")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    train_epochs, train_losses, partial_train_epochs, partial_train_losses = split_complete(epoch_rows, "train_loss")
    if train_epochs:
        ax.plot(train_epochs, train_losses, marker="o", color="#1f77b4", linewidth=2.0, label="train_loss")
    if partial_train_epochs:
        ax.plot(
            partial_train_epochs,
            partial_train_losses,
            marker="o",
            color="#1f77b4",
            linewidth=1.5,
            linestyle="--",
            label="train_loss_partial",
        )
    val_epochs = [float(row["epoch"]) for row in epoch_rows if "val_loss" in row and not math.isnan(float(row["val_loss"]))]
    val_losses = [float(row["val_loss"]) for row in epoch_rows if "val_loss" in row and not math.isnan(float(row["val_loss"]))]
    if val_epochs:
        ax.plot(val_epochs, val_losses, marker="s", color="#ff7f0e", linewidth=2.0, label="val_loss")
    ax.set_title("Epoch-Level Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2]
    train_epochs, train_nmse, partial_train_epochs, partial_train_nmse = split_complete(epoch_rows, "train_nmse")
    if train_epochs:
        ax.plot(train_epochs, train_nmse, marker="o", color="#2ca02c", linewidth=2.0, label="train_full_nmse")
    if partial_train_epochs:
        ax.plot(
            partial_train_epochs,
            partial_train_nmse,
            marker="o",
            color="#2ca02c",
            linewidth=1.5,
            linestyle="--",
            label="train_full_nmse_partial",
        )
    val_epochs = [float(row["epoch"]) for row in epoch_rows if "val_nmse" in row and not math.isnan(float(row["val_nmse"]))]
    val_nmse = [float(row["val_nmse"]) for row in epoch_rows if "val_nmse" in row and not math.isnan(float(row["val_nmse"]))]
    if val_epochs:
        ax.plot(val_epochs, val_nmse, marker="s", color="#9467bd", linewidth=2.0, label="val_nmse")
    ax.set_title("Epoch-Level Full-Model NMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMSE")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

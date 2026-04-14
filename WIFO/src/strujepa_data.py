from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_io import extract_first_array, load_mat_file


def _split_filename(split: str) -> str:
    normalized = str(split).strip().lower()
    mapping = {
        "train": "X_train.mat",
        "val": "X_val.mat",
        "valid": "X_val.mat",
        "validation": "X_val.mat",
        "test": "X_test.mat",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {sorted(mapping)}")
    return mapping[normalized]


def _preferred_keys(split: str) -> tuple[str, ...]:
    normalized = str(split).strip().lower()
    if normalized == "train":
        return ("X_train", "X", "X_val", "X_test")
    if normalized in {"val", "valid", "validation"}:
        return ("X_val", "X_valid", "X", "X_test")
    return ("X_test", "X_val", "X")


def load_channel_tensor(mat_path: str | Path, *, split: str) -> torch.Tensor:
    mat_data = load_mat_file(mat_path)
    array = extract_first_array(mat_data, _preferred_keys(split))
    complex_array = torch.tensor(np.array(array, dtype=complex)).unsqueeze(1)
    return torch.cat((complex_array.real, complex_array.imag), dim=1).float()


class ChannelTensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index].unsqueeze(0)


class SequentialLoader:
    def __init__(self, loaders: list[DataLoader], *, shuffle_loader_order: bool = False) -> None:
        self.loaders = list(loaders)
        self.shuffle_loader_order = bool(shuffle_loader_order)

    def __iter__(self):
        ordered = list(self.loaders)
        if self.shuffle_loader_order:
            random.shuffle(ordered)
        for loader in ordered:
            for batch in loader:
                yield batch

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)


def build_dataset(dataset_names: list[str], *, root: str | Path, split: str) -> list[Dataset]:
    root = Path(root)
    datasets: list[Dataset] = []
    filename = _split_filename(split)
    for name in dataset_names:
        path = root / name / filename
        tensor = load_channel_tensor(path, split=split)
        print(f"loaded split={split} dataset={name} shape={tuple(tensor.shape)}", flush=True)
        datasets.append(ChannelTensorDataset(tensor))
    if not datasets:
        raise ValueError("No datasets were requested")
    return datasets


def build_loader(
    dataset_names: list[str],
    *,
    root: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader | SequentialLoader:
    datasets = build_dataset(dataset_names, root=root, split=split)
    loaders = [
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=False,
        )
        for dataset in datasets
    ]
    if len(loaders) == 1:
        return loaders[0]
    return SequentialLoader(loaders, shuffle_loader_order=bool(shuffle))

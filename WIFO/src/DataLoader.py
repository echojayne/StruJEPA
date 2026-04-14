# coding=utf-8
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch as th
from torch.utils.data import Dataset

from data_io import extract_first_array, load_mat_file


class MyDataset(Dataset):
    def __init__(self, X_train):
        self.X_train = X_train

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx].unsqueeze(0)


def _load_channel_tensor(mat_path: str | Path, *, preferred_keys: tuple[str, ...]) -> torch.Tensor:
    mat_data = load_mat_file(mat_path)
    array = extract_first_array(mat_data, preferred_keys)
    complex_array = torch.tensor(np.array(array, dtype=complex)).unsqueeze(1)
    return torch.cat((complex_array.real, complex_array.imag), dim=1).float()


def data_load_single(args, dataset):  # 加载单个数据集
    folder_path_test = Path("../dataset") / dataset / "X_test.mat"
    X_test = _load_channel_tensor(folder_path_test, preferred_keys=("X_test", "X_val", "X"))
    test_data = MyDataset(X_test)
    batch_size = args.batch_size
    return th.utils.data.DataLoader(
        test_data,
        num_workers=32,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4,
    )


def data_load(args):
    test_data_all = []
    for dataset_name in args.dataset.split('*'):
        test_data = data_load_single(args, dataset_name)
        test_data_all.append(test_data)
    return test_data_all


def data_load_main(args):
    return data_load(args)

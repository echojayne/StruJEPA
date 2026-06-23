"""ImageNet data loading utilities for DINOv3 classification."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_imagenet_loaders(
    root: str | Path,
    *,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
) -> tuple[DataLoader, DataLoader, int]:
    root = Path(root).expanduser().resolve()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.ImageFolder(root / "train", transform=train_tf)
    val_set = datasets.ImageFolder(root / "val", transform=val_tf)
    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, int(len(train_set.classes))

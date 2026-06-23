"""Model helpers for StruJEPA on DINOv3 ImageNet classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "linear", "model_state_dict", "head"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise TypeError("unsupported head checkpoint format")


class DINOv3ImageNetClassifier(nn.Module):
    """Backbone plus linear head following the reference cls+patch-mean recipe."""

    def __init__(self, backbone: nn.Module, linear_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.representation_dim = int(backbone.embed_dim * 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone.forward_features(x)
        representation = torch.cat(
            (
                features["x_norm_clstoken"],
                features["x_norm_patchtokens"].mean(dim=1),
            ),
            dim=1,
        )
        logits = self.linear_head(representation)
        return {
            "logits": logits,
            "representation": representation,
            "cls_token": features["x_norm_clstoken"],
            "patch_tokens": features["x_norm_patchtokens"],
        }


def load_dinov3_backbone(
    *,
    repo_dir: str | Path,
    backbone_name: str,
    weight_path: str | Path | None,
) -> nn.Module:
    repo_dir = Path(repo_dir).expanduser().resolve()
    if weight_path:
        return torch.hub.load(
            str(repo_dir),
            backbone_name,
            source="local",
            weights=str(Path(weight_path).expanduser().resolve()),
        )
    return torch.hub.load(
        str(repo_dir),
        backbone_name,
        source="local",
        pretrained=False,
    )


def build_linear_head(
    *,
    backbone: nn.Module,
    num_classes: int,
    head_weight_path: str | Path | None = None,
) -> nn.Linear:
    linear_head = nn.Linear(int(backbone.embed_dim * 2), int(num_classes), bias=True)
    if head_weight_path:
        path = Path(head_weight_path).expanduser().resolve()
        if path.is_file():
            state_dict = _unwrap_state_dict(torch.load(path, map_location="cpu"))
            linear_head.load_state_dict(state_dict, strict=True)
    return linear_head

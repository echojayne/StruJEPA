"""Structural-mask encoding and representation alignment modules."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from elastic_method.core.structures import StructureMaskDescriptor


def descriptor_to_tensor(mask: StructureMaskDescriptor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    layer_indices = list(mask.selected_layer_indices)
    data = [
        float(mask.width_multiplier),
        float(mask.depth_multiplier),
        float(mask.total_layers),
        float(mask.active_num_heads),
        float(mask.active_ffn_dim),
        float(len(layer_indices)),
    ]
    data.extend(float(index) for index in layer_indices)
    return torch.tensor(data, device=device, dtype=dtype)


class StructuralMaskEncoder(nn.Module):
    """Encode a discrete structure mask descriptor into a dense embedding."""

    def __init__(self, input_dim: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
        )

    def forward(self, mask_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(mask_features)


class RepresentationAlignmentModule(nn.Module):
    """Predict a full-view representation from a masked-view representation."""

    def __init__(
        self,
        representation_dim: int,
        *,
        max_layers: int,
        mask_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.representation_dim = int(representation_dim)
        self.max_layers = int(max_layers)
        self.mask_input_dim = 6 + self.max_layers
        self.input_norm = nn.LayerNorm(self.representation_dim)
        self.target_norm = nn.LayerNorm(self.representation_dim)
        self.mask_encoder = StructuralMaskEncoder(self.mask_input_dim, embedding_dim=mask_embedding_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.representation_dim + mask_embedding_dim, self.representation_dim),
            nn.GELU(),
            nn.Linear(self.representation_dim, self.representation_dim),
        )

    def _encode_mask(self, mask: StructureMaskDescriptor, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        features = descriptor_to_tensor(mask, device=device, dtype=dtype)
        if features.numel() < self.mask_input_dim:
            features = torch.nn.functional.pad(features, (0, self.mask_input_dim - features.numel()))
        return self.mask_encoder(features.unsqueeze(0).expand(batch_size, -1))

    def normalize_target(self, representation: torch.Tensor) -> torch.Tensor:
        return self.target_norm(representation)

    def predict(self, representation: torch.Tensor, mask: StructureMaskDescriptor) -> torch.Tensor:
        normalized = self.input_norm(representation)
        mask_embedding = self._encode_mask(
            mask,
            batch_size=normalized.shape[0],
            device=normalized.device,
            dtype=normalized.dtype,
        )
        return self.predictor(torch.cat((normalized, mask_embedding), dim=-1))

    def forward(
        self,
        student_representation: torch.Tensor,
        teacher_representation: torch.Tensor,
        mask: StructureMaskDescriptor,
    ) -> dict[str, torch.Tensor]:
        teacher_target = self.normalize_target(teacher_representation.detach())
        predicted = self.predict(student_representation, mask)
        loss = torch.mean((predicted - teacher_target) ** 2)
        return {"loss": loss, "predicted_representation": predicted, "target_representation": teacher_target}


class StructuralMaskModule(nn.Module):
    """Default pluggable structural-mask representation alignment module."""

    def __init__(
        self,
        representation_dim: int,
        *,
        max_layers: int,
        mask_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.alignment = RepresentationAlignmentModule(
            representation_dim,
            max_layers=max_layers,
            mask_embedding_dim=mask_embedding_dim,
        )

    def forward(
        self,
        student_representation: torch.Tensor,
        teacher_representation: torch.Tensor,
        mask: StructureMaskDescriptor,
    ) -> dict[str, torch.Tensor]:
        return self.alignment(student_representation, teacher_representation, mask)

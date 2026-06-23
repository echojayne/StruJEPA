"""Training-time width missing-operator completion modules."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from elastic_method.core.runtime import get_runtime_state


PARAMETER_TYPES = (
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.proj",
    "ffn.fc1",
    "ffn.fc2",
)


@dataclass
class CompletionStageEpochs:
    warmup_completion: int = 0
    subnet_training: int = 0


@dataclass
class WidthOperatorCompletionConfig:
    enabled: bool = False
    mode: str = "width_operator_completion"
    stage_epochs: CompletionStageEpochs = field(default_factory=CompletionStageEpochs)
    depth_encoding: str = "gaussian_cdf"
    gaussian_mu: float = 0.5
    gaussian_sigma: float = 0.25
    lambda_weight: float = 1.0
    lambda_attn_residual: float = 1.0
    lambda_ffn_residual: float = 1.0
    predictor_hidden_dim: int = 32
    predictor_layers: int = 2
    predictor_num_heads: int = 4
    predictor_layout: str = "matrix_transformer_full"
    max_tokens_per_completion: int = 0
    use_layer_positional_encoding: bool = True


def parse_completion_config(payload: Any | None) -> WidthOperatorCompletionConfig:
    if isinstance(payload, WidthOperatorCompletionConfig):
        return payload
    if payload is None:
        return WidthOperatorCompletionConfig()
    if isinstance(payload, bool):
        return WidthOperatorCompletionConfig(enabled=bool(payload))
    if not isinstance(payload, dict):
        raise TypeError("method.completion must be a dict, bool, or WidthOperatorCompletionConfig")
    allowed_fields = {
        "enabled",
        "mode",
        "stage_epochs",
        "depth_encoding",
        "gaussian_mu",
        "gaussian_sigma",
        "lambda_weight",
        "lambda_attn_residual",
        "lambda_ffn_residual",
        "predictor_hidden_dim",
        "predictor_layers",
        "predictor_num_heads",
        "predictor_layout",
        "max_tokens_per_completion",
        "use_layer_positional_encoding",
        "use_lpe",
    }
    unsupported_fields = sorted(set(payload) - allowed_fields)
    if unsupported_fields:
        raise ValueError(f"unsupported completion fields: {unsupported_fields}")

    stage_payload = payload.get("stage_epochs", {})
    if isinstance(stage_payload, CompletionStageEpochs):
        stage_epochs = stage_payload
    elif isinstance(stage_payload, dict):
        unsupported_stage_fields = sorted(set(stage_payload) - {"warmup_completion", "subnet_training"})
        if unsupported_stage_fields:
            raise ValueError(f"unsupported completion stage fields: {unsupported_stage_fields}")
        stage_epochs = CompletionStageEpochs(
            warmup_completion=int(stage_payload.get("warmup_completion", 0)),
            subnet_training=int(stage_payload.get("subnet_training", 0)),
        )
    else:
        raise TypeError("method.completion.stage_epochs must be a dict or CompletionStageEpochs")

    config = WidthOperatorCompletionConfig(
        enabled=bool(payload.get("enabled", False)),
        mode=str(payload.get("mode", "width_operator_completion")),
        stage_epochs=stage_epochs,
        depth_encoding=str(payload.get("depth_encoding", "gaussian_cdf")),
        gaussian_mu=float(payload.get("gaussian_mu", 0.5)),
        gaussian_sigma=float(payload.get("gaussian_sigma", 0.25)),
        lambda_weight=float(payload.get("lambda_weight", 1.0)),
        lambda_attn_residual=float(payload.get("lambda_attn_residual", 1.0)),
        lambda_ffn_residual=float(payload.get("lambda_ffn_residual", 1.0)),
        predictor_hidden_dim=int(payload.get("predictor_hidden_dim", 32)),
        predictor_layers=int(payload.get("predictor_layers", 2)),
        predictor_num_heads=int(payload.get("predictor_num_heads", 4)),
        predictor_layout=str(payload.get("predictor_layout", "matrix_transformer_full")),
        max_tokens_per_completion=int(payload.get("max_tokens_per_completion", 0)),
        use_layer_positional_encoding=bool(
            payload.get(
                "use_layer_positional_encoding",
                payload.get("use_lpe", True),
            )
        ),
    )
    if config.enabled:
        if config.mode != "width_operator_completion":
            raise ValueError(
                "the current StruJEPA method only supports "
                "completion.mode='width_operator_completion'"
            )
        if config.predictor_layout != "matrix_transformer_full":
            raise ValueError(
                "the current StruJEPA method only supports "
                "completion.predictor_layout='matrix_transformer_full'"
            )
        if config.depth_encoding != "gaussian_cdf":
            raise ValueError(
                "the current StruJEPA method requires "
                "completion.depth_encoding='gaussian_cdf'"
            )
    return config


def _resolve_transformer_heads(d_model: int, requested_heads: int) -> int:
    requested = max(1, int(requested_heads))
    for heads in range(min(requested, int(d_model)), 0, -1):
        if int(d_model) % heads == 0:
            return heads
    return 1


class WidthMatrixTransformerPredictor(nn.Module):
    """Shared Transformer predictor for an entire missing width matrix."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.type_to_index = {name: idx for idx, name in enumerate(PARAMETER_TYPES)}
        self.depth_projection = nn.Linear(1, self.d_model)
        self.type_embedding = nn.Embedding(len(PARAMETER_TYPES), self.d_model)
        heads = _resolve_transformer_heads(self.d_model, int(num_heads))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=max(int(hidden_dim), self.d_model),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        shared_missing_rows: torch.Tensor,
        *,
        gaussian_scalar: torch.Tensor,
        parameter_type: str,
        use_layer_positional_encoding: bool = True,
    ) -> torch.Tensor:
        if shared_missing_rows.numel() == 0:
            return shared_missing_rows
        if shared_missing_rows.ndim != 2 or int(shared_missing_rows.shape[1]) != self.d_model:
            raise ValueError(
                "matrix-transformer completion expects missing weights with shape [missing_width, d_model]"
            )
        type_index = self.type_to_index[parameter_type]
        rows = int(shared_missing_rows.shape[0])
        tokens = shared_missing_rows
        gaussian = gaussian_scalar.to(
            device=shared_missing_rows.device,
            dtype=shared_missing_rows.dtype,
        )
        if bool(use_layer_positional_encoding):
            depth_feature = self.depth_projection(gaussian.reshape(1, 1)).expand(rows, -1)
            # LPE
            tokens = tokens + depth_feature
        type_feature = self.type_embedding(
            torch.tensor(type_index, device=shared_missing_rows.device, dtype=torch.long)
        ).to(dtype=shared_missing_rows.dtype)
        tokens = tokens + type_feature.view(1, -1).expand(rows, -1)
        encoded = self.encoder(tokens.unsqueeze(0)).squeeze(0)
        residual = self.output_projection(encoded)
        return shared_missing_rows + residual


def _infer_completion_d_model(shared_block: nn.Module) -> int:
    if hasattr(shared_block, "self_attn") and hasattr(shared_block.self_attn, "embed_dim"):
        return int(shared_block.self_attn.embed_dim)
    if hasattr(shared_block, "attn") and hasattr(shared_block.attn, "qkv"):
        return int(shared_block.attn.qkv.in_features)
    if hasattr(shared_block, "attn") and hasattr(shared_block.attn, "q"):
        return int(shared_block.attn.q.in_features)
    if hasattr(shared_block, "attention") and hasattr(shared_block.attention, "to_q"):
        return int(shared_block.attention.to_q.in_features)
    if hasattr(shared_block, "linear1"):
        return int(shared_block.linear1.in_features)
    if hasattr(shared_block, "mlp") and hasattr(shared_block.mlp, "fc1"):
        return int(shared_block.mlp.fc1.in_features)
    if hasattr(shared_block, "mlp") and hasattr(shared_block.mlp, "w1"):
        return int(shared_block.mlp.w1.in_features)
    raise ValueError("could not infer d_model for full-matrix width completion")


class WidthOperatorCompletionModule(nn.Module):
    """One shared trainable block plus the current full-matrix predictor."""

    def __init__(
        self,
        *,
        shared_block: nn.Module,
        total_layers: int,
        config: WidthOperatorCompletionConfig,
    ) -> None:
        super().__init__()
        self.shared_block = shared_block
        self.total_layers = int(total_layers)
        self.config = config
        predictor_layout = str(config.predictor_layout)
        if predictor_layout != "matrix_transformer_full":
            raise ValueError(f"unsupported width completion predictor_layout '{predictor_layout}'")
        self.predictor = WidthMatrixTransformerPredictor(
            d_model=_infer_completion_d_model(shared_block),
            hidden_dim=config.predictor_hidden_dim,
            num_layers=config.predictor_layers,
            num_heads=config.predictor_num_heads,
        )

    @classmethod
    def from_elastic_model(
        cls,
        model: nn.Module,
        config: WidthOperatorCompletionConfig,
    ) -> "WidthOperatorCompletionModule":
        if config.mode != "width_operator_completion":
            raise ValueError(f"unsupported completion mode '{config.mode}'")
        blocks = None
        stack_handle = getattr(model, "stack_handle", None)
        if stack_handle is not None:
            blocks = getattr(stack_handle, "blocks", None)
        if blocks is None:
            blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("width completion requires an elastic model with exposed blocks")
        metadata = getattr(model, "metadata", None)
        total_layers = int(getattr(metadata, "total_layers", len(blocks)))
        return cls(
            shared_block=deepcopy(blocks[0]),
            total_layers=total_layers,
            config=config,
        )

    def depth_encoding_scalar(
        self,
        *,
        layer_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.total_layers <= 1:
            normalized = 0.0
        else:
            normalized = float(layer_index) / float(self.total_layers - 1)
        sigma = max(float(self.config.gaussian_sigma), 1e-6)
        standardized = (normalized - float(self.config.gaussian_mu)) / sigma
        value = 0.5 * (1.0 + math.erf(standardized / math.sqrt(2.0)))
        return torch.tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _missing_as_rows(tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
        return tensor if int(dim) == 0 else tensor.transpose(0, 1)

    @staticmethod
    def _rows_as_missing(rows: torch.Tensor, *, dim: int) -> torch.Tensor:
        return rows if int(dim) == 0 else rows.transpose(0, 1)

    def _complete_tensor_matrix_full(
        self,
        *,
        active_slice: torch.Tensor,
        current_missing: torch.Tensor,
        shared_input: torch.Tensor,
        active_size: int,
        dim: int,
        depth_encoding: torch.Tensor,
        layer_index: int,
        parameter_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_input = torch.cat((active_slice, shared_input), dim=dim)
        full_rows = self._missing_as_rows(full_input, dim=dim)
        max_tokens = int(getattr(self.config, "max_tokens_per_completion", 0))
        if max_tokens > 0 and int(full_rows.shape[0]) > max_tokens:
            chunks = [
                self.predictor(
                    chunk,
                    gaussian_scalar=depth_encoding,
                    parameter_type=parameter_type,
                    use_layer_positional_encoding=self.config.use_layer_positional_encoding,
                )
                for chunk in full_rows.split(max_tokens, dim=0)
            ]
            predicted_full_rows = torch.cat(chunks, dim=0)
        else:
            predicted_full_rows = self.predictor(
                full_rows,
                gaussian_scalar=depth_encoding,
                parameter_type=parameter_type,
                use_layer_positional_encoding=self.config.use_layer_positional_encoding,
            )
        predicted_full = self._rows_as_missing(predicted_full_rows, dim=dim)
        predicted_missing = predicted_full.narrow(dim, int(active_size), int(current_missing.shape[dim]))
        predicted_missing_rows = self._missing_as_rows(predicted_missing, dim=dim)
        current_rows = self._missing_as_rows(current_missing, dim=dim)
        completion_loss = torch.mean((predicted_missing_rows - current_rows.detach()) ** 2)
        return predicted_missing, completion_loss

    def complete_tensor(
        self,
        current_weight: torch.Tensor,
        shared_weight: torch.Tensor,
        *,
        active_size: int,
        dim: int,
        layer_index: int,
        parameter_type: str,
    ) -> torch.Tensor:
        if current_weight.ndim != 2 or shared_weight.shape != current_weight.shape:
            raise ValueError("width completion only supports matching 2-D weight matrices")
        dim = int(dim)
        size = int(current_weight.shape[dim])
        active_size = max(1, min(int(active_size), size))
        if active_size >= size:
            return current_weight

        active_slice = current_weight.narrow(dim, 0, active_size)
        current_missing = current_weight.narrow(dim, active_size, size - active_size)
        shared_missing = shared_weight.narrow(dim, active_size, size - active_size)
        depth_encoding = self.depth_encoding_scalar(
            layer_index=int(layer_index),
            device=shared_missing.device,
            dtype=shared_missing.dtype,
        )
        predicted_missing, completion_loss = self._complete_tensor_matrix_full(
            active_slice=active_slice,
            current_missing=current_missing,
            shared_input=shared_missing,
            active_size=active_size,
            dim=dim,
            depth_encoding=depth_encoding,
            layer_index=int(layer_index),
            parameter_type=parameter_type,
        )
        runtime = get_runtime_state()
        if runtime is not None:
            runtime.completion_losses.append(completion_loss)
        return torch.cat((active_slice, predicted_missing), dim=dim)

    def complete_torch_encoder_attention(
        self,
        block: nn.Module,
        *,
        active_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        shared_attn = self.shared_block.self_attn
        embed_dim = int(block.self_attn.embed_dim)
        head_dim = int(block.self_attn.head_dim)
        active_dim = int(active_heads) * head_dim
        q_weight, k_weight, v_weight = block.self_attn.in_proj_weight.split(embed_dim, dim=0)
        sq_weight, sk_weight, sv_weight = shared_attn.in_proj_weight.split(embed_dim, dim=0)
        layer_index = int(block.layer_index)
        q_completed = self.complete_tensor(
            q_weight,
            sq_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.q",
        )
        k_completed = self.complete_tensor(
            k_weight,
            sk_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.k",
        )
        v_completed = self.complete_tensor(
            v_weight,
            sv_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.v",
        )
        out_completed = self.complete_tensor(
            block.self_attn.out_proj.weight,
            shared_attn.out_proj.weight,
            active_size=active_dim,
            dim=1,
            layer_index=layer_index,
            parameter_type="attn.proj",
        )
        return (
            torch.cat((q_completed, k_completed, v_completed), dim=0),
            block.self_attn.in_proj_bias,
            out_completed,
            block.self_attn.out_proj.bias,
        )

    def complete_torch_encoder_ffn(
        self,
        block: nn.Module,
        *,
        active_ffn_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        shared = self.shared_block
        layer_index = int(block.layer_index)
        fc1_weight = self.complete_tensor(
            block.linear1.weight,
            shared.linear1.weight,
            active_size=int(active_ffn_dim),
            dim=0,
            layer_index=layer_index,
            parameter_type="ffn.fc1",
        )
        fc2_weight = self.complete_tensor(
            block.linear2.weight,
            shared.linear2.weight,
            active_size=int(active_ffn_dim),
            dim=1,
            layer_index=layer_index,
            parameter_type="ffn.fc2",
        )
        return fc1_weight, block.linear1.bias, fc2_weight, block.linear2.bias

    def complete_wifo_attention(
        self,
        block: nn.Module,
        *,
        active_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        shared_attn = self.shared_block.attn
        head_dim = int(block.attn.q.out_features // block.attn.num_heads)
        active_dim = int(active_heads) * head_dim
        layer_index = int(block.layer_index)
        q_weight = self.complete_tensor(
            block.attn.q.weight,
            shared_attn.q.weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.q",
        )
        k_weight = self.complete_tensor(
            block.attn.k.weight,
            shared_attn.k.weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.k",
        )
        v_weight = self.complete_tensor(
            block.attn.v.weight,
            shared_attn.v.weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.v",
        )
        out_weight = self.complete_tensor(
            block.attn.proj.weight,
            shared_attn.proj.weight,
            active_size=active_dim,
            dim=1,
            layer_index=layer_index,
            parameter_type="attn.proj",
        )
        return (
            q_weight,
            block.attn.q.bias,
            k_weight,
            block.attn.k.bias,
            v_weight,
            block.attn.v.bias,
            out_weight,
            block.attn.proj.bias,
        )

    def complete_wifo_ffn(
        self,
        block: nn.Module,
        *,
        active_ffn_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        shared_mlp = self.shared_block.mlp
        layer_index = int(block.layer_index)
        fc1_weight = self.complete_tensor(
            block.mlp.fc1.weight,
            shared_mlp.fc1.weight,
            active_size=int(active_ffn_dim),
            dim=0,
            layer_index=layer_index,
            parameter_type="ffn.fc1",
        )
        fc2_weight = self.complete_tensor(
            block.mlp.fc2.weight,
            shared_mlp.fc2.weight,
            active_size=int(active_ffn_dim),
            dim=1,
            layer_index=layer_index,
            parameter_type="ffn.fc2",
        )
        return fc1_weight, block.mlp.fc1.bias, fc2_weight, block.mlp.fc2.bias

    def complete_perceiver_io_attention(
        self,
        block: nn.Module,
        *,
        active_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        shared_attn = self.shared_block.attention
        head_dim = int(block.attention.to_q.out_features // block.attention.heads)
        active_dim = int(active_heads) * head_dim
        layer_index = int(block.layer_index)
        k_weight, v_weight = block.attention.to_kv.weight.chunk(2, dim=0)
        shared_k_weight, shared_v_weight = shared_attn.to_kv.weight.chunk(2, dim=0)
        q_weight = self.complete_tensor(
            block.attention.to_q.weight,
            shared_attn.to_q.weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.q",
        )
        k_weight = self.complete_tensor(
            k_weight,
            shared_k_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.k",
        )
        v_weight = self.complete_tensor(
            v_weight,
            shared_v_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.v",
        )
        out_weight = self.complete_tensor(
            block.attention.to_out.weight,
            shared_attn.to_out.weight,
            active_size=active_dim,
            dim=1,
            layer_index=layer_index,
            parameter_type="attn.proj",
        )
        return (
            q_weight,
            block.attention.to_q.bias,
            k_weight,
            None,
            v_weight,
            None,
            out_weight,
            block.attention.to_out.bias,
        )

    def complete_perceiver_io_ffn(
        self,
        block: nn.Module,
        *,
        active_ffn_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        shared = self.shared_block
        layer_index = int(block.layer_index)
        value_weight, gate_weight = block.fc1.weight.chunk(2, dim=0)
        shared_value_weight, shared_gate_weight = shared.fc1.weight.chunk(2, dim=0)
        value_bias: torch.Tensor | None = None
        gate_bias: torch.Tensor | None = None
        if block.fc1.bias is not None:
            value_bias, gate_bias = block.fc1.bias.chunk(2, dim=0)
        value_completed = self.complete_tensor(
            value_weight,
            shared_value_weight,
            active_size=int(active_ffn_dim),
            dim=0,
            layer_index=layer_index,
            parameter_type="ffn.fc1",
        )
        gate_completed = self.complete_tensor(
            gate_weight,
            shared_gate_weight,
            active_size=int(active_ffn_dim),
            dim=0,
            layer_index=layer_index,
            parameter_type="ffn.fc1",
        )
        fc2_weight = self.complete_tensor(
            block.fc2.weight,
            shared.fc2.weight,
            active_size=int(active_ffn_dim),
            dim=1,
            layer_index=layer_index,
            parameter_type="ffn.fc2",
        )
        fc1_bias = torch.cat((value_bias, gate_bias), dim=0) if value_bias is not None and gate_bias is not None else None
        return torch.cat((value_completed, gate_completed), dim=0), fc1_bias, fc2_weight, block.fc2.bias

    def complete_dinov3_attention(
        self,
        block: nn.Module,
        *,
        active_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        shared_attn = self.shared_block.attn
        embed_dim = int(block.attn.qkv.in_features)
        head_dim = int(embed_dim // block.attn.num_heads)
        active_dim = int(active_heads) * head_dim
        q_weight, k_weight, v_weight = block.attn.qkv.weight.chunk(3, dim=0)
        sq_weight, sk_weight, sv_weight = shared_attn.qkv.weight.chunk(3, dim=0)
        layer_index = int(block.layer_index)
        q_completed = self.complete_tensor(
            q_weight,
            sq_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.q",
        )
        k_completed = self.complete_tensor(
            k_weight,
            sk_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.k",
        )
        v_completed = self.complete_tensor(
            v_weight,
            sv_weight,
            active_size=active_dim,
            dim=0,
            layer_index=layer_index,
            parameter_type="attn.v",
        )
        proj_completed = self.complete_tensor(
            block.attn.proj.weight,
            shared_attn.proj.weight,
            active_size=active_dim,
            dim=1,
            layer_index=layer_index,
            parameter_type="attn.proj",
        )
        return (
            torch.cat((q_completed, k_completed, v_completed), dim=0),
            block.attn.qkv.bias,
            proj_completed,
            block.attn.proj.bias,
        )

    def complete_dinov3_ffn(
        self,
        block: nn.Module,
        *,
        active_ffn_dim: int,
    ) -> tuple[Any, ...]:
        shared_mlp = self.shared_block.mlp
        layer_index = int(block.layer_index)
        if hasattr(block.mlp, "fc1") and hasattr(block.mlp, "fc2"):
            fc1_weight = self.complete_tensor(
                block.mlp.fc1.weight,
                shared_mlp.fc1.weight,
                active_size=int(active_ffn_dim),
                dim=0,
                layer_index=layer_index,
                parameter_type="ffn.fc1",
            )
            fc2_weight = self.complete_tensor(
                block.mlp.fc2.weight,
                shared_mlp.fc2.weight,
                active_size=int(active_ffn_dim),
                dim=1,
                layer_index=layer_index,
                parameter_type="ffn.fc2",
            )
            return "mlp", fc1_weight, block.mlp.fc1.bias, fc2_weight, block.mlp.fc2.bias
        if hasattr(block.mlp, "w1") and hasattr(block.mlp, "w2") and hasattr(block.mlp, "w3"):
            w1_weight = self.complete_tensor(
                block.mlp.w1.weight,
                shared_mlp.w1.weight,
                active_size=int(active_ffn_dim),
                dim=0,
                layer_index=layer_index,
                parameter_type="ffn.fc1",
            )
            w2_weight = self.complete_tensor(
                block.mlp.w2.weight,
                shared_mlp.w2.weight,
                active_size=int(active_ffn_dim),
                dim=0,
                layer_index=layer_index,
                parameter_type="ffn.fc1",
            )
            w3_weight = self.complete_tensor(
                block.mlp.w3.weight,
                shared_mlp.w3.weight,
                active_size=int(active_ffn_dim),
                dim=1,
                layer_index=layer_index,
                parameter_type="ffn.fc2",
            )
            return (
                "swiglu",
                w1_weight,
                block.mlp.w1.bias,
                w2_weight,
                block.mlp.w2.bias,
                w3_weight,
                block.mlp.w3.bias,
            )
        raise TypeError("unsupported DINOv3 FFN module for width completion")

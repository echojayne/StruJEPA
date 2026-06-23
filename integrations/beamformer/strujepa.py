"""StruJEPA integration for BeamFormer's PerceiverIO estimator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from elastic_method import ElasticizationSpec, elasticize_model
from elastic_method.core.structures import ForwardResult


BENCHMARK_ROOT = Path(
    os.environ.get(
        "BEAMFORMER_BENCHMARK_ROOT",
        "/mnt/dky/ai_ran_benchmarks/benchmarks/beam_management/beamformer",
    )
)
SOURCE_ROOT = Path(os.environ.get("BEAMFORMER_SOURCE_ROOT", BENCHMARK_ROOT / "source"))
DEFAULT_CSI_ROOT = BENCHMARK_ROOT / "assets" / "csi-dataset" / "homeoffice-communication-28G-csi"
DEFAULT_WEIGHT_ROOT = BENCHMARK_ROOT / "assets" / "original_weights" / "final"


def ensure_beamformer_source(path: str | Path = SOURCE_ROOT) -> Path:
    source = Path(path)
    if not (source / "beamformer").is_dir():
        raise FileNotFoundError(f"BeamFormer source not found: {source}")
    text = str(source)
    if text not in sys.path:
        sys.path.insert(0, text)
    return source


def build_setting(
    *,
    csi_root: str | Path = DEFAULT_CSI_ROOT,
    weight_root: str | Path = DEFAULT_WEIGHT_ROOT,
    source_root: str | Path = SOURCE_ROOT,
) -> SimpleNamespace:
    ensure_beamformer_source(source_root)
    from configs.submodules import ARN_model, assumption, dataset, estimator, generator

    csi_root = Path(csi_root)
    weight_root = Path(weight_root)
    ds = dataset.homeoffice_communication_28g()
    ds.train_data_path = str(csi_root / "t16x16_r2x1_train")
    ds.test_data_path = str(csi_root / "t16x16_r2x1_test_small")
    return SimpleNamespace(
        name="strujepa_beamformer",
        dataset=ds,
        assumption=assumption.beam64(),
        estimator=estimator.PerceiverIO(estimator_pretrained_model=str(weight_root / "estimator.pth")),
        generator=generator.parametric_generator(generator_pretrained_model=str(weight_root / "generator.pth")),
        arn_model=ARN_model.typical_ARN(ARN_model_pretrained_model=str(weight_root / "arn_model.pth")),
    )


def build_loaders(
    setting: SimpleNamespace,
    *,
    batch_size: int,
    num_workers: int = 0,
    train_split: str = "t16x16_r2x1_train",
    val_split: str = "t16x16_r2x1_test_small",
    train_limit: int = 0,
    val_limit: int = 0,
) -> tuple[Any, Any]:
    ensure_beamformer_source()
    from beamformer.dataset import csi_dataset

    root = Path(setting.dataset.train_data_path).parent
    train_dataset = csi_dataset(
        str(root / train_split),
        M_tx=setting.dataset.M_tx,
        N_tx=setting.dataset.N_tx,
        M_rx=setting.dataset.M_rx,
        N_rx=setting.dataset.N_rx,
        add_noise=setting.dataset.add_noise,
        snr_min=setting.dataset.snr_min,
        subcarrier_bw=setting.dataset.subcarrier_spacing,
    )
    val_dataset = csi_dataset(
        str(root / val_split),
        M_tx=setting.dataset.M_tx,
        N_tx=setting.dataset.N_tx,
        M_rx=setting.dataset.M_rx,
        N_rx=setting.dataset.N_rx,
        add_noise=setting.dataset.add_noise,
        snr_min=setting.dataset.snr_min,
        subcarrier_bw=setting.dataset.subcarrier_spacing,
    )
    train_dataset.dataset_path = [path for path in train_dataset.dataset_path if str(path).endswith(".mat")]
    val_dataset.dataset_path = [path for path in val_dataset.dataset_path if str(path).endswith(".mat")]
    if train_limit > 0:
        train_dataset.dataset_path = train_dataset.dataset_path[: int(train_limit)]
    if val_limit > 0:
        val_dataset.dataset_path = val_dataset.dataset_path[: int(val_limit)]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


def load_original_estimator(setting: SimpleNamespace) -> torch.nn.Module:
    ensure_beamformer_source()
    from beamformer.modules import TransformerModel

    model = TransformerModel(setting.estimator)
    checkpoint = Path(setting.estimator.estimator_pretrained_model)
    if checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


def load_original_generator(setting: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    ensure_beamformer_source()
    from beamformer.weight_generator import ParametricGenerator

    generator = ParametricGenerator(setting.assumption.sample_num, setting.dataset.M, setting.dataset.N)
    checkpoint = Path(setting.generator.generator_pretrained_model)
    if checkpoint.is_file():
        generator.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    generator.to(device)
    generator.eval()
    generator.requires_grad_(False)
    return generator


def elasticize_beamformer_estimator(
    estimator: torch.nn.Module,
    *,
    width_multipliers: tuple[float, ...],
    depth_multipliers: tuple[float, ...],
    copy_model: bool = False,
):
    spec = ElasticizationSpec(
        stack_path="transformer.layers",
        block_family="perceiver_io",
        width_multipliers=tuple(float(value) for value in width_multipliers),
        depth_multipliers=tuple(float(value) for value in depth_multipliers),
        width_only_epochs=0,
    )
    return elasticize_model(estimator, spec, copy_model=copy_model), spec


@dataclass
class BeamFormerBatchCache:
    query_pos_encoding: torch.Tensor | None = None
    batch_size: int = 0
    device: torch.device | None = None


class BeamFormerTaskAdapter:
    """Generate BeamFormer RSS supervision and train the estimator only."""

    def __init__(
        self,
        setting: SimpleNamespace,
        *,
        generator: torch.nn.Module,
        device: torch.device,
        use_lpe: bool = True,
    ) -> None:
        ensure_beamformer_source()
        from beamformer.dataset import load_data_process

        self.setting = setting
        self.generator = generator
        self.device = torch.device(device)
        # This flag is kept for run metadata. BeamFormer's native sample/query
        # positional encodings are part of the task input and must stay enabled;
        # the StruJEPA LPE ablation is controlled in the completion module.
        self.use_lpe = bool(use_lpe)
        self.dp = load_data_process(setting, device=self.device)
        self.cache = BeamFormerBatchCache()

    def _query_pos_encoding(self, batch_size: int) -> torch.Tensor:
        cached = self.cache
        if cached.query_pos_encoding is not None and cached.batch_size == int(batch_size) and cached.device == self.device:
            return cached.query_pos_encoding
        query_pos = self.dp.generate_query_position_encoding(batch_size=int(batch_size)).to(dtype=torch.float32)
        cached.query_pos_encoding = query_pos
        cached.batch_size = int(batch_size)
        cached.device = self.device
        return query_pos

    def prepare_batch(self, batch: Any, device: torch.device) -> dict[str, Any]:
        ensure_beamformer_source()
        from beamformer.utils import scale_in_last_dim
        from beamformer.weight_generator import transform_weights

        csi = batch[0].to(device=device, non_blocking=True)
        batch_size = int(csi.shape[0])
        z_dim = int(self.setting.assumption.sample_num * self.setting.dataset.M * self.setting.dataset.N)
        with torch.no_grad():
            z = torch.randn(batch_size, z_dim, device=device)
            raw_weights = self.generator(z)
            weights, _ = transform_weights(raw_weights)
            sample_pos_enc = self.dp.generate_sample_position_encoding(weights).to(dtype=torch.float32)
            query_pos_enc = self._query_pos_encoding(batch_size)
            sample_rss = self.dp.generate_sample_rss(csi, weights).to(dtype=torch.float32)
            query_rss = self.dp.generate_query_rss(csi).to(dtype=torch.float32)
            scale = torch.max(sample_rss, dim=1, keepdim=True).values.clamp_min(1e-12)
            sample_rss = sample_rss / scale
            query_rss = query_rss / scale
            target, _ = scale_in_last_dim(query_rss)
        return {
            "model_args": (sample_rss, sample_pos_enc, query_pos_enc),
            "model_kwargs": {},
            "target": target,
            "query_rss": query_rss,
            "scale": scale,
        }

    def batch_size(self, batch: Any) -> int:
        return int(batch["target"].shape[0])

    @staticmethod
    def _prediction(result: ForwardResult) -> torch.Tensor:
        output = result.model_output
        if isinstance(output, tuple):
            return output[0]
        return output

    def compute_supervised_loss(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        return F.mse_loss(self._prediction(result), batch["target"])

    def extract_alignment_view(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        del batch
        return self._prediction(result)

    def compute_output_alignment_loss(
        self,
        result: ForwardResult,
        reference_result: ForwardResult,
        batch: Any,
    ) -> torch.Tensor:
        del batch
        return F.mse_loss(self._prediction(result), self._prediction(reference_result).detach().clone())

    def compute_metrics(self, result: ForwardResult, batch: Any) -> dict[str, float]:
        prediction = self._prediction(result)
        target = batch["target"]
        mse = F.mse_loss(prediction, target)
        mae = torch.mean(torch.abs(prediction - target))
        pred_idx = torch.argmax(prediction, dim=1)
        target_at_pred = torch.gather(batch["query_rss"], 1, pred_idx.view(-1, 1)).squeeze(1).clamp_min(1e-12)
        oracle = torch.max(batch["query_rss"], dim=1).values.clamp_min(1e-12)
        rss_loss_db = 10.0 * torch.log10(oracle) - 10.0 * torch.log10(target_at_pred)
        return {
            "spectrum_mse": float(mse.detach().item()),
            "spectrum_mae": float(mae.detach().item()),
            "rss_loss_db": float(torch.mean(rss_loss_db).detach().item()),
        }

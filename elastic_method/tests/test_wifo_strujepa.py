from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method import MethodConfig
from integrations.wifo.benchmark_paths import ensure_wifo_source_path
from integrations.wifo.elastic_wifo import elasticize_wifo
from integrations.wifo.strujepa_recipe_trainer import WiFoStruJEPATrainer
from integrations.wifo.strujepa_wifo import WiFoTaskAdapter

WIFO_SRC = ensure_wifo_source_path()
if not (WIFO_SRC / "model.py").exists():
    raise unittest.SkipTest(f"WiFo benchmark source not found at {WIFO_SRC}")
from model import WiFo_model  # noqa: E402


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        size="tiny",
        patch_size=4,
        t_patch_size=4,
        pos_emb="SinCos_3D",
        no_qkv_bias=0,
    )


class TinyWiFoDataset(Dataset[torch.Tensor]):
    def __init__(self, length: int = 4) -> None:
        self.samples = torch.randn(length, 1, 2, 4, 8, 8)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


class WiFoStruJEPATest(unittest.TestCase):
    def test_seeded_forward_is_deterministic(self) -> None:
        torch.manual_seed(7)
        model = WiFo_model(args=make_args()).eval()
        inputs = torch.randn(2, 1, 2, 4, 8, 8)
        with torch.inference_mode():
            first = model(inputs, mask_ratio=0.5, mask_strategy="random", seed=2026)
            second = model(inputs, mask_ratio=0.5, mask_strategy="random", seed=2026)
        self.assertTrue(torch.allclose(first[0], second[0], atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(first[1], second[1], atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(torch.view_as_real(first[2]), torch.view_as_real(second[2]), atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(torch.view_as_real(first[3]), torch.view_as_real(second[3]), atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.equal(first[4], second[4]))

    def test_task_adapter_expands_remote_task_mix(self) -> None:
        task_adapter = WiFoTaskAdapter(
            task_specs="random:0.85,temporal:0.5,fre:0.5",
            base_seed=11,
        )
        prepared = task_adapter.prepare_batch(torch.randn(2, 1, 2, 4, 8, 8), device=torch.device("cpu"))
        tasks = task_adapter.expand_task_batches(prepared, epoch=3, batch_index=7)
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0]["model_kwargs"]["mask_strategy"], "random")
        self.assertAlmostEqual(tasks[0]["model_kwargs"]["mask_ratio"], 0.85)
        self.assertEqual(tasks[1]["model_kwargs"]["mask_strategy"], "temporal")
        self.assertAlmostEqual(tasks[1]["model_kwargs"]["mask_ratio"], 0.5)
        self.assertEqual(tasks[2]["model_kwargs"]["mask_strategy"], "fre")
        self.assertEqual(tasks[0]["model_kwargs"]["seed"], 11 * 1_000_000 + 3 * 10_000 + 7 * 100)
        self.assertEqual(tasks[2]["model_kwargs"]["seed"], 11 * 1_000_000 + 3 * 10_000 + 7 * 100 + 2)

    def test_strujepa_trainer_smoke_runs(self) -> None:
        torch.manual_seed(17)
        elastic = elasticize_wifo(
            WiFo_model(args=make_args()),
            width_multipliers=(1.0, 0.5, 0.125),
            depth_multipliers=(1.0, 0.5),
            copy_model=True,
        )
        task_adapter = WiFoTaskAdapter(
            task_specs="random:0.85,temporal:0.5,fre:0.5",
            base_seed=99,
        )
        trainer = WiFoStruJEPATrainer(
            elastic,
            task_adapter,
            spec=elastic.spec,
            config=MethodConfig(supervised_weight=0.75),
            random_subnets_per_batch=1,
            sampling_seed=99,
            validate_every=1,
            subnet_sampling_mode="anchor_random",
            objective_mode="full_plus_mean_subnets",
            device="cpu",
        )
        loader = DataLoader(TinyWiFoDataset(), batch_size=2, shuffle=False)
        history = trainer.fit(loader, epochs=1)
        self.assertEqual(len(history), 1)
        self.assertIn("train_loss", history[-1])
        self.assertIn("train_nmse", history[-1])
        self.assertIn("train_task_loss", history[-1])
        sampled = trainer._sample_subnets(trainer._enumerate_subnets(epoch=1), epoch=1, batch_index=1)
        sampled_pairs = {(subnet.width_multiplier, subnet.depth_multiplier) for subnet in sampled}
        self.assertIn((1.0, 1.0), sampled_pairs)
        self.assertIn((0.5, 0.5), sampled_pairs)
        self.assertIn((0.125, 0.5), sampled_pairs)
        self.assertIn((0.125, 1.0), sampled_pairs)
        self.assertIn((1.0, 0.5), sampled_pairs)

    def test_width_completion_two_stage_smoke_runs(self) -> None:
        torch.manual_seed(57)
        elastic = elasticize_wifo(
            WiFo_model(args=make_args()),
            width_multipliers=(1.0, 0.5),
            depth_multipliers=(1.0, 0.5),
            copy_model=True,
        )
        task_adapter = WiFoTaskAdapter(
            task_specs="random:0.5",
            base_seed=57,
        )
        trainer = WiFoStruJEPATrainer(
            elastic,
            task_adapter,
            spec=elastic.spec,
            config=MethodConfig(
                supervised_weight=1.0,
                completion={
                    "enabled": True,
                    "mode": "width_operator_completion",
                    "depth_encoding": "gaussian_cdf",
                    "predictor_layout": "matrix_transformer_full",
                    "stage_epochs": {
                        "warmup_completion": 1,
                        "subnet_training": 1,
                    },
                    "lambda_weight": 0.05,
                    "lambda_attn_residual": 0.05,
                    "lambda_ffn_residual": 0.05,
                },
            ),
            random_subnets_per_batch=0,
            sampling_seed=57,
            validate_every=1,
            subnet_sampling_mode="all",
            objective_mode="full_plus_mean_subnets",
            device="cpu",
        )
        loader = DataLoader(TinyWiFoDataset(length=1), batch_size=1, shuffle=False)
        history = trainer.fit(loader, epochs=2)
        self.assertIn("train_weight_completion_loss", history[0])
        self.assertIn("train_attn_residual_loss", history[0])
        self.assertNotIn("train_weight_completion_loss", history[1])
        self.assertIsNotNone(trainer.completion_module)
        self.assertFalse(hasattr(trainer.completion_module, "shared_blocks"))

    def test_scale_anchor_sampling_uses_large_mid_small_only(self) -> None:
        torch.manual_seed(58)
        elastic = elasticize_wifo(
            WiFo_model(args=make_args()),
            width_multipliers=(1.0, 0.75, 0.5, 0.25, 0.125),
            depth_multipliers=(1.0, 0.5, 0.333333, 0.25, 0.166667),
            copy_model=True,
        )
        task_adapter = WiFoTaskAdapter(
            task_specs="random:0.5",
            base_seed=58,
        )
        trainer = WiFoStruJEPATrainer(
            elastic,
            task_adapter,
            spec=elastic.spec,
            config=MethodConfig(),
            random_subnets_per_batch=0,
            sampling_seed=58,
            validate_every=1,
            subnet_sampling_mode="scale_anchors",
            objective_mode="full_plus_mean_subnets",
            device="cpu",
        )
        sampled = trainer._sample_subnets(trainer._enumerate_subnets(epoch=1), epoch=1, batch_index=1)
        self.assertEqual(
            [(subnet.width_multiplier, subnet.depth_multiplier) for subnet in sampled],
            [(1.0, 1.0), (0.5, 0.333333), (0.125, 0.166667)],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
WIFO_SRC = ROOT / "WIFO" / "src"
if str(WIFO_SRC) not in sys.path:
    sys.path.insert(0, str(WIFO_SRC))

from elastic_method import MethodConfig
from elastic_wifo import elasticize_wifo
from model import WiFo_model
from strujepa_recipe_trainer import WiFoStruJEPATrainer
from strujepa_wifo import WiFoStruJEPACallback


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

    def test_callback_expands_remote_task_mix(self) -> None:
        callback = WiFoStruJEPACallback(
            representation_dim=64,
            task_specs="random:0.85,temporal:0.5,fre:0.5",
            base_seed=11,
        )
        prepared = callback.prepare_batch(torch.randn(2, 1, 2, 4, 8, 8), device=torch.device("cpu"))
        tasks = callback.expand_task_batches(prepared, epoch=3, batch_index=7)
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
        callback = WiFoStruJEPACallback(
            representation_dim=elastic.model.embed_dim,
            task_specs="random:0.85,temporal:0.5,fre:0.5",
            base_seed=99,
        )
        trainer = WiFoStruJEPATrainer(
            elastic,
            callback,
            spec=elastic.spec,
            config=MethodConfig(
                use_ema_full_view=True,
                enable_output_alignment=True,
                enable_repr_alignment=True,
                supervised_weight=0.75,
                lambda_output=1.5,
                lambda_repr=0.25,
            ),
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
        self.assertIn("train_output_alignment_loss", history[-1])
        self.assertIn("train_repr_alignment_loss", history[-1])
        sampled = trainer._sample_subnets(trainer._enumerate_subnets(epoch=1), epoch=1, batch_index=1)
        sampled_pairs = {(subnet.width_multiplier, subnet.depth_multiplier) for subnet in sampled}
        self.assertIn((1.0, 1.0), sampled_pairs)
        self.assertIn((0.5, 0.5), sampled_pairs)
        self.assertIn((0.125, 0.5), sampled_pairs)
        self.assertIn((0.125, 1.0), sampled_pairs)
        self.assertIn((1.0, 0.5), sampled_pairs)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from elastic_method import ElasticizationSpec, MethodConfig, elasticize_model
from elastic_method.method.trainer import StruJEPATrainer
from elastic_method.tasks.regression import MeanPooledRegressionTaskAdapter


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, length: int = 8, *, seq_len: int = 4, dim: int = 16) -> None:
        self.inputs = torch.randn(length, seq_len, dim)
        self.targets = self.inputs.mean(dim=(1, 2), keepdim=True)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.inputs[index], "targets": self.targets[index]}


class TinyTorchEncoderModel(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 3, heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).mean(dim=1))


def make_trainer(completion_overrides: dict[str, object] | None = None) -> StruJEPATrainer:
    elastic = elasticize_model(
        TinyTorchEncoderModel(),
        ElasticizationSpec(
            stack_path="encoder",
            block_family="torch_encoder",
            width_multipliers=(1.0, 0.5, 0.25),
            depth_multipliers=(1.0, 0.5),
            width_only_epochs=0,
        ),
    )
    completion = {
        "enabled": True,
        "mode": "width_operator_completion",
        "stage_epochs": {"warmup_completion": 1, "subnet_training": 1},
        "lambda_weight": 0.1,
        "lambda_attn_residual": 0.1,
        "lambda_ffn_residual": 0.1,
        "depth_encoding": "gaussian_cdf",
        "predictor_layout": "matrix_transformer_full",
        "predictor_hidden_dim": 32,
        "predictor_layers": 1,
        "predictor_num_heads": 4,
    }
    if completion_overrides:
        completion.update(completion_overrides)
    return StruJEPATrainer(
        elastic,
        MeanPooledRegressionTaskAdapter(),
        spec=elastic.spec,
        config=MethodConfig(completion=completion),
        device="cpu",
    )


def make_elastic_model() -> nn.Module:
    return elasticize_model(
        TinyTorchEncoderModel(),
        ElasticizationSpec(
            stack_path="encoder",
            block_family="torch_encoder",
            width_multipliers=(1.0, 0.5, 0.25),
            depth_multipliers=(1.0, 0.5),
            width_only_epochs=0,
        ),
    )


def zero_completion_residual(module: nn.Module) -> None:
    nn.init.zeros_(module.predictor.output_projection.weight)
    nn.init.zeros_(module.predictor.output_projection.bias)


class WidthOperatorCompletionTest(unittest.TestCase):
    def test_reference_model_is_frozen_initial_full_view(self) -> None:
        torch.manual_seed(101)
        trainer = make_trainer()
        self.assertIsNotNone(trainer.completion_module)
        self.assertIsNotNone(trainer.reference_model)
        self.assertEqual(trainer.completion_module.total_layers, 3)
        batch = next(iter(DataLoader(TinyDataset(length=2), batch_size=2)))
        prepared = trainer.task_adapter.prepare_batch(batch, trainer.device)
        trainer.model.eval()
        with torch.inference_mode():
            current = trainer.model(
                *prepared["model_args"],
                width_multiplier=1.0,
                depth_multiplier=1.0,
                **prepared["model_kwargs"],
            )
            reference = trainer.reference_model(
                *prepared["model_args"],
                width_multiplier=1.0,
                depth_multiplier=1.0,
                **prepared["model_kwargs"],
            )
        self.assertTrue(torch.allclose(current.model_output, reference.model_output, atol=1e-6, rtol=1e-6))
        self.assertFalse(any(parameter.requires_grad for parameter in trainer.reference_model.parameters()))

    def test_active_slice_is_current_and_missing_slice_uses_shared_predictor(self) -> None:
        torch.manual_seed(202)
        module = make_trainer().completion_module
        self.assertIsNotNone(module)
        zero_completion_residual(module)
        current = torch.arange(256, dtype=torch.float32).view(16, 16)
        shared = torch.arange(1000, 1256, dtype=torch.float32).view(16, 16)
        completed = module.complete_tensor(
            current,
            shared,
            active_size=8,
            dim=0,
            layer_index=1,
            parameter_type="attn.q",
        )
        self.assertTrue(torch.equal(completed[:8], current[:8]))
        self.assertTrue(torch.allclose(completed[8:], shared[8:]))

    def test_gaussian_cdf_depth_encoding_is_monotonic(self) -> None:
        module = make_trainer({"gaussian_mu": 0.5, "gaussian_sigma": 0.2}).completion_module
        self.assertIsNotNone(module)
        values = [
            float(module.depth_encoding_scalar(layer_index=index, device=torch.device("cpu"), dtype=torch.float32))
            for index in range(module.total_layers)
        ]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0])

    def test_lpe_ablation_skips_depth_projection(self) -> None:
        module = make_trainer({"use_layer_positional_encoding": False}).completion_module
        self.assertIsNotNone(module)
        current = torch.arange(256, dtype=torch.float32).view(16, 16)
        shared = torch.arange(1000, 1256, dtype=torch.float32).view(16, 16)
        with mock.patch.object(
            module.predictor.depth_projection,
            "forward",
            side_effect=AssertionError("LPE-disabled completion must not use depth projection"),
        ):
            completed = module.complete_tensor(
                current,
                shared,
                active_size=8,
                dim=0,
                layer_index=1,
                parameter_type="attn.q",
            )
        self.assertEqual(tuple(completed.shape), tuple(current.shape))

    def test_two_stage_subnet_policy(self) -> None:
        trainer = make_trainer()
        warmup = trainer._stage_subnets(stage="completion_warmup", epoch=1)
        final = trainer._stage_subnets(stage="subnet_training", epoch=2)
        self.assertEqual({subnet.depth_multiplier for subnet in warmup}, {1.0})
        self.assertIn(0.5, {subnet.depth_multiplier for subnet in final})
        self.assertEqual(trainer._stage(1), "completion_warmup")
        self.assertEqual(trainer._stage(2), "subnet_training")

    def test_training_subnet_sampling_preserves_full_and_smallest_views(self) -> None:
        trainer = make_trainer()
        trainer.config.subnets_per_batch = 4
        subnets = trainer._stage_subnets(stage="subnet_training", epoch=2)
        sampled = trainer._select_subnets_for_batch(
            subnets,
            epoch=2,
            batch_index=3,
            train=True,
        )
        self.assertEqual(len(sampled), 4)
        self.assertIn((1.0, 1.0), {(s.width_multiplier, s.depth_multiplier) for s in sampled})
        self.assertIn((0.25, 0.5), {(s.width_multiplier, s.depth_multiplier) for s in sampled})
        validation = trainer._select_subnets_for_batch(
            subnets,
            epoch=2,
            batch_index=3,
            train=False,
        )
        self.assertEqual(validation, subnets)

    def test_adaptive_focus_adds_previous_worst_subnet(self) -> None:
        trainer = make_trainer()
        trainer.config.subnets_per_batch = 4
        trainer.config.adaptive_focus_subnets = 1
        trainer._update_adaptive_subnet_focus(
            {
                "val_subnet_w1_d1_rss_loss_db": 0.5,
                "val_subnet_w1_d0.5_rss_loss_db": 3.0,
                "val_subnet_w0.25_d0.5_rss_loss_db": 1.0,
            }
        )
        subnets = trainer._stage_subnets(stage="subnet_training", epoch=2)
        sampled = trainer._select_subnets_for_batch(
            subnets,
            epoch=3,
            batch_index=4,
            train=True,
        )
        labels = {
            trainer._subnet_label(subnet.width_multiplier, subnet.depth_multiplier)
            for subnet in sampled
        }
        self.assertIn("w1_d0.5", labels)
        self.assertIn("w0.25_d0.5", labels)
        self.assertIn("w1_d1", labels)

    def test_adaptive_focus_can_rank_by_gap_to_baseline_target(self) -> None:
        trainer = make_trainer()
        trainer.config.subnets_per_batch = 4
        trainer.config.adaptive_focus_subnets = 1
        trainer.config.subnet_rss_targets_db = {
            "w1_d0.5": 2.9,
            "w0.5_d0.5": 0.5,
        }
        trainer._update_adaptive_subnet_focus(
            {
                "val_subnet_w1_d1_rss_loss_db": 0.5,
                "val_subnet_w1_d0.5_rss_loss_db": 3.0,
                "val_subnet_w0.5_d0.5_rss_loss_db": 1.0,
            }
        )
        self.assertEqual(trainer._priority_subnet_labels, ["w0.5_d0.5"])

    def test_subnet_target_gap_includes_requested_margin(self) -> None:
        trainer = make_trainer()
        trainer.config.subnet_rss_targets_db = {
            "w1_d1": 0.6,
            "w0.5_d1": 0.9,
        }
        trainer.config.target_margin_db = 0.05
        self.assertAlmostEqual(trainer._target_gap_db("w1_d1", 0.58), 0.03)
        self.assertAlmostEqual(trainer._target_gap_db("w0.5_d1", 0.8), -0.05)
        self.assertIsNone(trainer._target_gap_db("w0.25_d1", 1.0))

    def test_hardest_subnet_loss_and_full_view_weight_are_applied(self) -> None:
        trainer = make_trainer()
        trainer.config.full_view_weight = 2.0
        trainer.config.hardest_subnet_weight = 0.5
        full = torch.tensor(1.0)
        subnets = [torch.tensor(2.0), torch.tensor(4.0)]
        mean_loss = (2.0 * full + subnets[0] + subnets[1]) / 4.0
        expected = 0.5 * mean_loss + 0.5 * torch.tensor(4.0)
        self.assertTrue(torch.equal(trainer._combine_task_losses(full, subnets), expected))

    def test_target_gap_weights_prioritize_lagging_subnets(self) -> None:
        trainer = make_trainer()
        trainer.config.adaptive_focus_subnets = 2
        trainer.config.target_gap_loss_weight = 2.0
        trainer.config.full_view_weight = 1.0
        trainer.config.hardest_subnet_weight = 0.0
        trainer.config.subnet_rss_targets_db = {
            "w1_d1": 0.7,
            "w0.5_d1": 0.7,
            "w0.25_d1": 0.7,
        }
        record = {
            "val_subnet_w1_d1_rss_loss_db": 0.6,
            "val_subnet_w0.5_d1_rss_loss_db": 1.2,
            "val_subnet_w0.25_d1_rss_loss_db": 0.95,
        }
        trainer._update_adaptive_subnet_focus(record)
        self.assertEqual(
            trainer._target_loss_multipliers,
            {
                "w1_d1": 1.0,
                "w0.5_d1": 3.0,
                "w0.25_d1": 2.0,
            },
        )
        self.assertEqual(
            record["target_loss_multipliers"],
            trainer._target_loss_multipliers,
        )
        combined = trainer._combine_task_losses(
            torch.tensor(1.0),
            [torch.tensor(2.0), torch.tensor(3.0)],
            full_label="w1_d1",
            subnet_labels=["w0.5_d1", "w0.25_d1"],
        )
        self.assertAlmostEqual(float(combined), 13.0 / 6.0, places=6)

    def test_initial_target_weights_apply_before_first_validation(self) -> None:
        elastic = make_elastic_model()
        trainer = StruJEPATrainer(
            elastic,
            MeanPooledRegressionTaskAdapter(),
            spec=elastic.spec,
            config=MethodConfig(
                initial_subnet_loss_multipliers={
                    "w1_d1": 1.0,
                    "w0.5_d1": 3.0,
                },
                initial_priority_subnet_labels=["w0.5_d1"],
            ),
            device="cpu",
        )
        self.assertEqual(trainer._target_loss_multipliers["w0.5_d1"], 3.0)
        self.assertEqual(trainer._priority_subnet_labels, ["w0.5_d1"])
        combined = trainer._combine_task_losses(
            torch.tensor(1.0),
            [torch.tensor(2.0)],
            full_label="w1_d1",
            subnet_labels=["w0.5_d1"],
        )
        self.assertAlmostEqual(float(combined), 7.0 / 4.0, places=6)

    def test_reference_model_remains_frozen_during_warmup(self) -> None:
        torch.manual_seed(404)
        trainer = make_trainer()
        loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
        reference_before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.reference_model.named_parameters()
        }
        trainer.run_epoch(loader, epoch=1, train=True)
        self.assertTrue(
            all(
                torch.allclose(parameter.detach(), reference_before[name])
                for name, parameter in trainer.reference_model.named_parameters()
            )
        )

    def test_historical_completion_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported completion fields"):
            make_trainer({"elastic_update": 1})

    def test_two_stage_smoke_logs_expected_losses(self) -> None:
        torch.manual_seed(505)
        trainer = make_trainer()
        loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
        history = trainer.fit(loader, epochs=2)
        self.assertIn("train_weight_completion_loss", history[0])
        self.assertIn("train_attn_residual_loss", history[0])
        self.assertIn("train_ffn_residual_loss", history[0])
        self.assertIn("train_task_loss", history[1])
        self.assertIn("train_output_alignment_loss", history[1])
        self.assertIn("train_repr_alignment_loss", history[1])
        self.assertNotIn("train_weight_completion_loss", history[1])
        self.assertGreaterEqual(history[-1]["train_loss"], 0.0)

    def test_completion_warmup_uses_no_task_losses(self) -> None:
        torch.manual_seed(5051)
        trainer = make_trainer()
        loader = DataLoader(TinyDataset(length=4), batch_size=4, shuffle=False)
        with (
            mock.patch.object(
                trainer.task_adapter,
                "compute_supervised_loss",
                side_effect=AssertionError("completion warmup must not use supervised loss"),
            ),
            mock.patch.object(
                trainer,
                "_output_alignment_loss",
                side_effect=AssertionError("completion warmup must not use output alignment loss"),
            ),
        ):
            metrics = trainer.run_stage_epoch(
                loader,
                stage="completion_warmup",
                epoch=1,
                train=False,
            )
        self.assertGreaterEqual(metrics["loss"], 0.0)
        self.assertEqual(metrics["supervised_loss"], 0.0)
        self.assertEqual(metrics["output_alignment_loss"], 0.0)

    def test_output_warmup_uses_alignment_without_supervised_loss(self) -> None:
        torch.manual_seed(5052)
        elastic = make_elastic_model()
        trainer = StruJEPATrainer(
            elastic,
            MeanPooledRegressionTaskAdapter(),
            spec=elastic.spec,
            config=MethodConfig(
                completion=False,
                lambda_output=1.0,
                subnets_per_batch=3,
            ),
            device="cpu",
        )
        loader = DataLoader(TinyDataset(length=4), batch_size=4, shuffle=False)
        with mock.patch.object(
            trainer.task_adapter,
            "compute_supervised_loss",
            side_effect=AssertionError("output warmup must not use supervised loss"),
        ):
            metrics = trainer.run_stage_epoch(
                loader,
                stage="output_warmup",
                epoch=1,
                train=False,
            )
        self.assertGreaterEqual(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["output_alignment_loss"], 0.0)
        self.assertEqual(metrics["supervised_loss"], 0.0)

    def test_task_only_subnet_stage_skips_reference_forward(self) -> None:
        torch.manual_seed(506)
        trainer = make_trainer()
        trainer.config.lambda_output = 0.0
        trainer.config.lambda_repr = 0.0
        loader = DataLoader(TinyDataset(length=4), batch_size=4, shuffle=False)
        with mock.patch.object(
            trainer,
            "_reference_forward",
            side_effect=AssertionError("task-only stage must not run the reference model"),
        ):
            metrics = trainer.run_stage_epoch(
                loader,
                stage="subnet_training",
                epoch=2,
                train=False,
            )
        self.assertEqual(metrics["output_alignment_loss"], 0.0)
        self.assertEqual(metrics["repr_alignment_loss"], 0.0)
        self.assertGreaterEqual(metrics["task_loss"], 0.0)

    def test_external_optimizer_is_extended_with_completion_parameters(self) -> None:
        torch.manual_seed(606)
        elastic = make_elastic_model()
        optimizer = torch.optim.AdamW(elastic.parameters(), lr=1e-4)
        trainer = StruJEPATrainer(
            elastic,
            MeanPooledRegressionTaskAdapter(),
            spec=elastic.spec,
            config=MethodConfig(
                completion={
                    "enabled": True,
                    "mode": "width_operator_completion",
                    "stage_epochs": {"warmup_completion": 1, "subnet_training": 1},
                    "depth_encoding": "gaussian_cdf",
                    "predictor_layout": "matrix_transformer_full",
                }
            ),
            optimizer=optimizer,
            device="cpu",
        )
        self.assertIsNotNone(trainer.completion_module)
        optimizer_params = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        completion_params = {id(parameter) for parameter in trainer.completion_module.parameters()}
        self.assertTrue(completion_params.issubset(optimizer_params))

    def test_fit_until_converged_runs_warmup_before_subnet_training(self) -> None:
        torch.manual_seed(707)
        elastic = make_elastic_model()
        trainer = StruJEPATrainer(
            elastic,
            MeanPooledRegressionTaskAdapter(),
            spec=elastic.spec,
            config=MethodConfig(
                completion={
                    "enabled": True,
                    "mode": "width_operator_completion",
                    "stage_epochs": {"warmup_completion": 1, "subnet_training": 1},
                    "depth_encoding": "gaussian_cdf",
                    "predictor_layout": "matrix_transformer_full",
                },
                stage_convergence={
                    "enabled": True,
                    "completion_warmup": {
                        "metric": "val_loss",
                        "min_epochs": 1,
                        "max_epochs": 1,
                        "patience": 0,
                    },
                    "subnet_training": {
                        "metric": "val_task_loss",
                        "min_epochs": 1,
                        "max_epochs": 1,
                        "patience": 0,
                    },
                },
            ),
            device="cpu",
        )
        loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
        history, reports = trainer.fit_until_converged(loader, val_loader=loader)
        self.assertEqual([record["stage"] for record in history], ["completion_warmup", "subnet_training"])
        self.assertEqual([report["stage"] for report in reports], ["completion_warmup", "subnet_training"])
        self.assertTrue(all(report["converged"] for report in reports))

    def test_fit_until_converged_resumes_stage_epoch_without_replaying_it(self) -> None:
        torch.manual_seed(808)
        trainer = make_trainer()
        trainer.config.stage_convergence = {
            "enabled": True,
            "completion_warmup": {
                "metric": "val_loss",
                "min_epochs": 2,
                "max_epochs": 2,
                "patience": 0,
            },
            "subnet_training": {
                "metric": "val_task_loss",
                "min_epochs": 1,
                "max_epochs": 1,
                "patience": 0,
            },
        }
        loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
        initial_history = [
            {
                "epoch": 1.0,
                "stage": "completion_warmup",
                "stage_epoch": 1.0,
                "val_loss": 1.0,
                "convergence_metric": "val_loss",
                "convergence_metric_value": 1.0,
                "convergence_best_metric": 1.0,
                "convergence_best_stage_epoch": 1.0,
                "convergence_stale_epochs": 0.0,
                "convergence_improved": True,
            }
        ]
        history, reports = trainer.fit_until_converged(
            loader,
            val_loader=loader,
            initial_history=initial_history,
        )
        self.assertEqual(
            [(record["stage"], record["stage_epoch"]) for record in history],
            [
                ("completion_warmup", 1.0),
                ("completion_warmup", 2.0),
                ("subnet_training", 1.0),
            ],
        )
        self.assertEqual([report["stage"] for report in reports], ["completion_warmup", "subnet_training"])

    def test_fit_until_converged_can_run_only_requested_stage(self) -> None:
        torch.manual_seed(909)
        trainer = make_trainer()
        trainer.config.stage_convergence = {
            "enabled": True,
            "stages": ["subnet_training"],
            "subnet_training": {
                "metric": "val_task_loss",
                "min_epochs": 1,
                "max_epochs": 1,
                "patience": 0,
            },
        }
        loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
        history, reports = trainer.fit_until_converged(loader, val_loader=loader)
        self.assertEqual([record["stage"] for record in history], ["subnet_training"])
        self.assertEqual([report["stage"] for report in reports], ["subnet_training"])


if __name__ == "__main__":
    unittest.main()

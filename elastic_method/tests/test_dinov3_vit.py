from __future__ import annotations

import os
import unittest
from pathlib import Path
import sys

import torch

from elastic_method import ElasticizationSpec, elasticize_model
from integrations.dinov3_imagenet.modeling import DINOv3ImageNetClassifier


class DinoV3VitElasticizationTest(unittest.TestCase):
    def test_full_view_matches_original(self) -> None:
        repo_dir = Path(os.environ.get("DINOV3_ROOT", Path.home() / "dinov3"))
        if not repo_dir.is_dir():
            self.skipTest("DINOV3_ROOT is not available")
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        from dinov3.models.vision_transformer import DinoVisionTransformer

        torch.manual_seed(17)
        backbone = DinoVisionTransformer(
            img_size=32,
            patch_size=16,
            embed_dim=64,
            depth=2,
            num_heads=4,
            ffn_ratio=4.0,
            qkv_bias=True,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=0,
            mask_k_bias=False,
        )
        backbone.init_weights()
        head = torch.nn.Linear(int(backbone.embed_dim * 2), 10)
        classifier = DINOv3ImageNetClassifier(backbone, head)
        elastic = elasticize_model(
            classifier,
            ElasticizationSpec(
                stack_path="backbone.blocks",
                block_family="dinov3_vit",
                width_multipliers=(1.0, 0.5),
                depth_multipliers=(1.0, 0.5),
            ),
        )
        inputs = torch.randn(2, 3, 32, 32)
        classifier.eval()
        elastic.eval()
        with torch.inference_mode():
            baseline = classifier(inputs)["logits"]
            wrapped = elastic(inputs, width_multiplier=1.0, depth_multiplier=1.0, return_encoder_state=True)
        self.assertTrue(torch.allclose(baseline, wrapped.model_output["logits"], atol=1e-6, rtol=1e-6))
        self.assertEqual(wrapped.encoder_state.shape[0], 2)

    def test_full_view_matches_original_with_swiglu(self) -> None:
        repo_dir = Path(os.environ.get("DINOV3_ROOT", Path.home() / "dinov3"))
        if not repo_dir.is_dir():
            self.skipTest("DINOV3_ROOT is not available")
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        from dinov3.models.vision_transformer import DinoVisionTransformer

        torch.manual_seed(23)
        backbone = DinoVisionTransformer(
            img_size=32,
            patch_size=16,
            embed_dim=96,
            depth=2,
            num_heads=6,
            ffn_ratio=3.0,
            qkv_bias=False,
            norm_layer="layernormbf16",
            ffn_layer="swiglu64",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=0,
            mask_k_bias=False,
        )
        backbone.init_weights()
        head = torch.nn.Linear(int(backbone.embed_dim * 2), 10)
        classifier = DINOv3ImageNetClassifier(backbone, head)
        elastic = elasticize_model(
            classifier,
            ElasticizationSpec(
                stack_path="backbone.blocks",
                block_family="dinov3_vit",
                width_multipliers=(1.0, 0.5),
                depth_multipliers=(1.0, 0.5),
            ),
        )
        inputs = torch.randn(2, 3, 32, 32)
        classifier.eval()
        elastic.eval()
        with torch.inference_mode():
            baseline = classifier(inputs)["logits"]
            wrapped = elastic(inputs, width_multiplier=1.0, depth_multiplier=1.0, return_encoder_state=True)
        self.assertTrue(torch.allclose(baseline, wrapped.model_output["logits"], atol=1e-6, rtol=1e-6))
        self.assertEqual(wrapped.encoder_state.shape[0], 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
WIFO_SRC = ROOT / "WIFO" / "src"
if str(WIFO_SRC) not in sys.path:
    sys.path.insert(0, str(WIFO_SRC))

from elastic_wifo import elasticize_wifo  # noqa: E402
from model import WiFo_model  # noqa: E402


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        size="tiny",
        patch_size=4,
        t_patch_size=4,
        pos_emb="SinCos_3D",
        no_qkv_bias=0,
    )


class WifoVitElasticizationTest(unittest.TestCase):
    def test_full_view_matches_original(self) -> None:
        torch.manual_seed(11)
        baseline_model = WiFo_model(args=make_args()).eval()
        elastic_source = deepcopy(baseline_model).eval()
        elastic = elasticize_wifo(
            elastic_source,
            width_multipliers=(1.0, 0.5),
            depth_multipliers=(1.0, 0.5),
        ).eval()
        inputs = torch.randn(2, 1, 2, 12, 16, 96)
        baseline_inputs = [sample for sample in inputs]
        torch.manual_seed(123)
        with torch.inference_mode():
            baseline = baseline_model(baseline_inputs, mask_ratio=0.5, mask_strategy="random")
        torch.manual_seed(123)
        with torch.inference_mode():
            wrapped = elastic(
                inputs,
                mask_ratio=0.5,
                mask_strategy="random",
                width_multiplier=1.0,
                depth_multiplier=1.0,
                return_encoder_state=True,
            )
        self.assertTrue(torch.allclose(baseline[0], wrapped.model_output[0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(baseline[1], wrapped.model_output[1], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.view_as_real(baseline[2]), torch.view_as_real(wrapped.model_output[2]), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.view_as_real(baseline[3]), torch.view_as_real(wrapped.model_output[3]), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(baseline[4], wrapped.model_output[4]))
        self.assertEqual(tuple(wrapped.encoder_state.shape), (2, 144, 64))


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace

import torch

from models.feature_regularization import AnisotropicDiffusion, TotalVariation
from losses.spatial_regularization import SpatialRegularizationLoss


class FeatureRegularizationTest(unittest.TestCase):
    def test_diffusion_preserves_constant_features(self):
        features = torch.ones(2, 4, 5, 5)
        output = AnisotropicDiffusion(num_steps=3)(features)
        torch.testing.assert_close(output, features)

    def test_diffusion_preserves_semantic_edge(self):
        features = torch.zeros(1, 2, 3, 4)
        features[:, 0, :, :2] = 1.0
        features[:, 1, :, 2:] = 1.0
        features[:, :, 1, 0] += 0.2

        weak = AnisotropicDiffusion(tau=0.2, sigma=0.1)(features)
        strong = AnisotropicDiffusion(tau=0.2, sigma=10.0)(features)
        self.assertLess(weak[0, 0, 1, 0], features[0, 0, 1, 0])
        weak_change = (weak[:, :, :, 1] - features[:, :, :, 1]).abs().mean()
        strong_change = (strong[:, :, :, 1] - features[:, :, :, 1]).abs().mean()
        self.assertLess(weak_change, strong_change)

    def test_diffusion_is_differentiable(self):
        features = torch.randn(2, 3, 4, 4, requires_grad=True)
        AnisotropicDiffusion(num_steps=2)(features).square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_total_variation_matches_definition(self):
        features = torch.tensor([[[[0.0, 1.0], [2.0, 4.0]]]], requires_grad=True)
        loss = TotalVariation(reduction="sum")(features)
        self.assertAlmostEqual(loss.item(), 8.0)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_total_variation_edge_cases(self):
        tv = TotalVariation()
        self.assertEqual(tv(torch.ones(2, 3, 4, 4)).item(), 0.0)
        single = torch.randn(2, 3, 1, 1, requires_grad=True)
        tv(single).backward()
        self.assertIsNotNone(single.grad)

    def test_invalid_diffusion_parameters(self):
        for kwargs in ({"num_steps": 0}, {"tau": 0.3}, {"sigma": 0.0}):
            with self.assertRaises(ValueError):
                AnisotropicDiffusion(**kwargs)

    def test_tv_loss_is_only_active_for_tv_model(self):
        output = {"feat_0": torch.randn(2, 3, 4, 4, requires_grad=True)}
        cfg = SimpleNamespace(
            feature_regularization="total_variation",
            head_idx_patch=0,
            weight=0.05,
            reduction="mean",
        )
        losses = SpatialRegularizationLoss(cfg)(outputs_stu=output)
        self.assertEqual(losses[0]["name"], "total_variation")
        self.assertEqual(losses[0]["weight"], 0.05)

        cfg.feature_regularization = "anisotropic_diffusion"
        self.assertEqual(SpatialRegularizationLoss(cfg)(outputs_stu=output), [])


if __name__ == "__main__":
    unittest.main()

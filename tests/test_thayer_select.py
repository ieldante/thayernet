"""Contract tests for coordinate prompts and the Thayer-Select interface."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.coordinate_prompt import gaussian_coordinate_prompt
from src.models_thayer_select import ThayerSelectNet
from src.recoverability import (
    RecoverabilityThresholds,
    bounded_heteroscedastic_gaussian_nll,
    operational_recoverable_label_v04,
    recoverability_bce_loss,
    selective_risk_curve,
)


class CoordinatePromptTests(unittest.TestCase):
    def test_prompt_peaks_at_integer_requested_coordinate(self) -> None:
        coordinates = torch.tensor([[7.0, 11.0]])
        prompt = gaussian_coordinate_prompt(coordinates, 20, 24, sigma_pixels=2.0)
        flat_index = int(torch.argmax(prompt[0, 0]))
        y, x = divmod(flat_index, 24)
        self.assertEqual((x, y), (7, 11))
        self.assertEqual(float(prompt[0, 0, y, x]), 1.0)

    def test_empty_and_wrong_prompts_are_representable(self) -> None:
        coordinates = torch.tensor([[4.0, 5.0], [14.0, 2.0], [19.0, 2.0]])
        prompt = gaussian_coordinate_prompt(
            coordinates, 16, 16, valid=torch.tensor([False, True, True])
        )
        self.assertEqual(int(torch.count_nonzero(prompt[0])), 0)
        self.assertGreater(float(prompt[1].max()), 0.0)
        self.assertEqual(int(torch.count_nonzero(prompt[2])), 0)


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("MPS is required; CPU fallback is prohibited")
        cls.device = torch.device("mps")

    def test_output_shapes_and_separate_queries(self) -> None:
        torch.manual_seed(7)
        model = ThayerSelectNet(base_channels=4).to(self.device)
        image = torch.randn(1, 3, 32, 32, device=self.device)
        prompt_a = gaussian_coordinate_prompt(
            torch.tensor([[10.0, 10.0]], device=self.device), 32, 32
        )
        prompt_b = gaussian_coordinate_prompt(
            torch.tensor([[22.0, 19.0]], device=self.device), 32, 32
        )
        output_a = model(image, prompt_a)
        output_b = model(image, prompt_b)
        self.assertEqual(output_a["reconstruction"].shape, (1, 3, 32, 32))
        self.assertEqual(output_a["log_variance"].shape, (1, 3, 32, 32))
        self.assertEqual(output_a["recoverability"].shape, (1, 1))
        self.assertFalse(torch.equal(output_a["reconstruction"], output_b["reconstruction"]))

    def test_losses_are_finite_and_differentiable(self) -> None:
        torch.manual_seed(8)
        model = ThayerSelectNet(base_channels=4).to(self.device)
        image = torch.randn(2, 3, 16, 16, device=self.device)
        prompt = gaussian_coordinate_prompt(
            torch.tensor([[5.0, 5.0], [10.0, 11.0]], device=self.device), 16, 16
        )
        target = torch.randn_like(image)
        output = model(image, prompt)
        reconstruction_loss = bounded_heteroscedastic_gaussian_nll(
            output["reconstruction"], target, output["log_variance"]
        )
        score_labels = torch.tensor([[1.0], [0.0]], device=self.device)
        score_loss = recoverability_bce_loss(
            output["recoverability"], score_labels
        )
        loss = reconstruction_loss + score_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        score_gradients = [
            parameter.grad
            for parameter in model.recoverability_head.parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in score_gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in score_gradients)
        )

    def test_variance_cannot_improve_nll_without_bound(self) -> None:
        prediction = torch.zeros(1, 1, 1, 1, device=self.device)
        target = torch.ones_like(prediction)
        at_cap = bounded_heteroscedastic_gaussian_nll(
            prediction, target, torch.tensor([[[[4.0]]]], device=self.device)
        )
        far_above_cap = bounded_heteroscedastic_gaussian_nll(
            prediction, target, torch.tensor([[[[1000.0]]]], device=self.device)
        )
        self.assertEqual(float(at_cap), float(far_above_cap))

    def test_selective_risk_uses_score_and_evaluation_loss_only(self) -> None:
        curve = selective_risk_curve(np.array([0.2, 0.9]), np.array([4.0, 1.0]))
        self.assertTrue(np.isnan(curve["risk"][0]))
        np.testing.assert_allclose(curve["risk"][1:], [1.0, 2.5])
        np.testing.assert_allclose(curve["coverage"], [0.0, 0.5, 1.0])

    def test_selective_risk_ties_are_emitted_as_complete_groups(self) -> None:
        curve = selective_risk_curve(
            np.array([0.9, 0.9, 0.2]), np.array([1.0, 3.0, 8.0])
        )
        np.testing.assert_allclose(curve["coverage"], [0.0, 2 / 3, 1.0])
        self.assertTrue(np.isnan(curve["risk"][0]))
        np.testing.assert_allclose(curve["risk"][1:], [2.0, 4.0])
        np.testing.assert_allclose(curve["threshold"][1:], [0.9, 0.2])

    def test_selective_risk_refuses_silent_nonfinite_exclusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            selective_risk_curve(np.array([0.9, np.nan]), np.array([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            selective_risk_curve(np.array([0.9, 0.1]), np.array([0.0, np.inf]))

    def test_operational_label_rejects_negative_physical_errors(self) -> None:
        valid = dict(
            evaluation_valid=True,
            usable_band_count=1,
            affected_region_nmse=0.01,
            max_usable_band_flux_fraction_error=0.01,
            centroid_error_pixels=0.1,
            max_applicable_false_subtraction_fraction=0.01,
        )
        self.assertTrue(operational_recoverable_label_v04(**valid))
        policy = RecoverabilityThresholds()
        at_boundary = {
            **valid,
            "affected_region_nmse": policy.affected_region_nmse,
            "max_usable_band_flux_fraction_error": policy.whole_source_flux_fraction_error,
            "centroid_error_pixels": policy.centroid_error_pixels,
            "max_applicable_false_subtraction_fraction": (
                policy.core_false_subtraction_fraction
            ),
        }
        self.assertTrue(operational_recoverable_label_v04(**at_boundary))
        for field, boundary in (
            ("affected_region_nmse", policy.affected_region_nmse),
            (
                "max_usable_band_flux_fraction_error",
                policy.whole_source_flux_fraction_error,
            ),
            ("centroid_error_pixels", policy.centroid_error_pixels),
            (
                "max_applicable_false_subtraction_fraction",
                policy.core_false_subtraction_fraction,
            ),
        ):
            with self.subTest(boundary_field=field):
                self.assertFalse(
                    operational_recoverable_label_v04(
                        **{**valid, field: np.nextafter(boundary, np.inf)}
                    )
                )
        for field in (
            "affected_region_nmse",
            "max_usable_band_flux_fraction_error",
            "centroid_error_pixels",
            "max_applicable_false_subtraction_fraction",
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    operational_recoverable_label_v04(**{**valid, field: -0.01})
                )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            operational_recoverable_label_v04(
                **valid,
                thresholds=RecoverabilityThresholds(affected_region_nmse=-1.0),
            )
        for invalid_count in (True, 1.5, 4, np.nan):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaisesRegex(ValueError, "integer in"):
                    operational_recoverable_label_v04(
                        **{**valid, "usable_band_count": invalid_count}
                    )


if __name__ == "__main__":
    unittest.main()

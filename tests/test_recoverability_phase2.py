"""Boundary and safety tests for Phase-II prompt semantics and contracts."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.models_thayer_select import ThayerSelectNet
from src.prompt_semantics import PromptSemantics, QueryClass, associate_prompt
from src.recoverability import PHASE2_CONTRACTS, phase2_contract_success
from scripts.thayer_select_recoverability_common import add_actionable_acceptance_labels
from scripts.thayer_select_recoverability_common import read_csv, write_csv_union_fresh


class PromptSemanticsPhase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = np.asarray([[10.0, 10.0], [18.0, 10.0]])
        self.policy = PromptSemantics()

    def test_exact_source_and_alternate_are_valid_requests(self) -> None:
        for index in (0, 1):
            result = associate_prompt(self.sources, self.sources[index], image_shape=(60, 60))
            self.assertEqual(result.query_class, QueryClass.VALID_SOURCE)
            self.assertEqual(result.matched_index, index)

    def test_perturbed_unique_request(self) -> None:
        result = associate_prompt(self.sources, np.asarray([12.0, 10.0]), image_shape=(60, 60))
        self.assertEqual(result.query_class, QueryClass.PERTURBED_VALID)
        self.assertEqual(result.matched_index, 0)

    def test_null_and_edge_null(self) -> None:
        for prompt in (np.asarray([40.0, 40.0]), np.asarray([0.0, 59.0])):
            self.assertEqual(associate_prompt(self.sources, prompt, image_shape=(60, 60)).query_class, QueryClass.NULL_SOURCE)

    def test_equal_distance_is_ambiguous(self) -> None:
        result = associate_prompt(self.sources, np.asarray([14.0, 10.0]), image_shape=(60, 60))
        self.assertEqual(result.query_class, QueryClass.AMBIGUOUS_SOURCE)
        self.assertIsNone(result.matched_index)

    def test_radius_boundary_and_just_outside(self) -> None:
        one = np.asarray([[10.0, 10.0]])
        at = associate_prompt(one, np.asarray([14.0, 10.0]), image_shape=(60, 60))
        outside = associate_prompt(one, np.asarray([14.0 + 1e-9, 10.0]), image_shape=(60, 60))
        self.assertEqual(at.query_class, QueryClass.PERTURBED_VALID)
        self.assertEqual(outside.query_class, QueryClass.NULL_SOURCE)

    def test_out_of_frame_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            associate_prompt(self.sources, np.asarray([-0.1, 10.0]), image_shape=(60, 60))


class ReliabilityContractPhase2Tests(unittest.TestCase):
    def test_null_outcome_success_is_not_global_actionable_success(self) -> None:
        metrics = {"evaluation_valid": True, "hallucination": False, "catastrophic_failure": False}
        labeled = add_actionable_acceptance_labels(metrics, QueryClass.NULL_SOURCE)
        self.assertEqual(labeled["moderate_success"], 1)
        self.assertEqual(labeled["moderate_actionable_success"], 0)

    def test_heterogeneous_metric_rows_use_union_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_csv_union_fresh(path, [{"scene_id": "null", "hallucination": 0}, {"scene_id": "valid", "g_relative_flux_error": 0.1}])
            rows = read_csv(path)
        self.assertEqual(rows[0]["scene_id"], "null")
        self.assertEqual(rows[1]["g_relative_flux_error"], "0.1")

    def test_moderate_boundaries_and_prohibitions(self) -> None:
        contract = PHASE2_CONTRACTS["moderate"]
        metrics = {
            "evaluation_valid": True,
            "normalized_rmse": contract.max_normalized_rmse,
            "max_relative_flux_error": contract.max_relative_flux_error,
            "max_color_error_mag": contract.max_color_error_mag,
            "centroid_error_pixels": contract.max_centroid_error_pixels,
            "source_confusion": False,
            "catastrophic_failure": False,
        }
        self.assertTrue(phase2_contract_success(metrics, QueryClass.VALID_SOURCE, contract))
        self.assertFalse(phase2_contract_success({**metrics, "source_confusion": True}, QueryClass.VALID_SOURCE, contract))
        self.assertFalse(phase2_contract_success(metrics, QueryClass.AMBIGUOUS_SOURCE, contract))
        self.assertTrue(phase2_contract_success({"evaluation_valid": True, "hallucination": False, "catastrophic_failure": False}, QueryClass.NULL_SOURCE, contract))
        self.assertFalse(phase2_contract_success({"evaluation_valid": True, "hallucination": True, "catastrophic_failure": False}, QueryClass.NULL_SOURCE, contract))

    def test_uncertainty_and_probability_outputs_are_bounded(self) -> None:
        if not torch.backends.mps.is_available():
            self.skipTest("MPS required by campaign policy")
        device = torch.device("mps")
        model = ThayerSelectNet(base_channels=4, min_log_variance=-8.0, max_log_variance=2.0).to(device)
        output = model(
            torch.randn(2, 3, 16, 16, device=device),
            torch.rand(2, 1, 16, 16, device=device),
        )
        self.assertTrue(torch.all(output["log_variance"] >= -8.0))
        self.assertTrue(torch.all(output["log_variance"] <= 2.0))
        self.assertTrue(torch.all((output["recoverability"] >= 0.0) & (output["recoverability"] <= 1.0)))
        self.assertTrue(torch.all((output["no_source_probability"] >= 0.0) & (output["no_source_probability"] <= 1.0)))


if __name__ == "__main__":
    unittest.main()

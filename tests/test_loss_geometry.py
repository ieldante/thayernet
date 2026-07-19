from __future__ import annotations

import unittest

import numpy as np
import torch

from src.loss_geometry import (
    TOP_LEVEL_WEIGHTS,
    canonical_configurations,
    exact_batch_objective,
    scene_loss_terms,
    source_light_transfer,
)
from src.models_two_expert_decoder import source_sum


class LossGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.rows = [
            {"kind": "ordinary", "near_collision_pair_id": ""},
            {"kind": "ordinary", "near_collision_pair_id": ""},
            {"kind": "near_collision", "near_collision_pair_id": "pair-1"},
            {"kind": "near_collision", "near_collision_pair_id": "pair-1"},
        ]
        self.targets = rng.normal(size=(4, 2, 2, 6, 5, 4)).astype(np.float32)
        self.counts = np.asarray([[1, 1], [1, 1], [2, 2], [2, 2]], dtype=np.int64)
        self.targets[:2, :, 1] = self.targets[:2, :, 0]
        self.trained = rng.normal(size=self.targets.shape).astype(np.float32)
        self.blend = rng.normal(size=(4, 3, 5, 4)).astype(np.float32)

    def test_canonical_configuration_contract(self) -> None:
        configs = canonical_configurations(self.targets, self.counts, self.trained, self.rows)
        self.assertEqual(len(configs), 13)
        ordinary_indices, exact = configs["O1_EXACT_TRUTH_DUPLICATED"]
        self.assertEqual(ordinary_indices.tolist(), [0, 1])
        np.testing.assert_array_equal(exact[:, :, 0], exact[:, :, 1])
        ambiguous_indices, approved = configs["A1_EXACT_APPROVED_SET"]
        np.testing.assert_array_equal(approved, self.targets[ambiguous_indices])

    def test_source_light_transfer_preserves_sum(self) -> None:
        base = torch.from_numpy(self.targets.copy())
        for fraction in (-0.5, -0.1, 0.0, 0.25, 0.5):
            moved = source_light_transfer(base, fraction)
            self.assertTrue(torch.allclose(source_sum(moved), source_sum(base), atol=1e-7))

    def test_scene_accounting_reproduces_batch_objective(self) -> None:
        output = torch.from_numpy(self.trained)
        target = torch.from_numpy(self.targets)
        counts = torch.from_numpy(self.counts)
        blend = torch.from_numpy(self.blend)
        scene = scene_loss_terms(output, target, counts, blend, self.rows, np.arange(4))
        batch = exact_batch_objective(output, target, counts, blend, self.rows)
        self.assertLessEqual(abs(float(scene["total"].mean()) - float(batch["total"])), 2e-6)

    def test_exact_truth_has_zero_target_terms(self) -> None:
        configs = canonical_configurations(self.targets, self.counts, self.trained, self.rows)
        for name in ("O1_EXACT_TRUTH_DUPLICATED", "A1_EXACT_APPROVED_SET"):
            indices, values = configs[name]
            terms = scene_loss_terms(
                torch.from_numpy(values), torch.from_numpy(self.targets[indices]),
                torch.from_numpy(self.counts[indices]), torch.from_numpy(self.blend[indices]),
                self.rows, indices,
            )
            for term in ("requested_reconstruction", "companion_reconstruction", "target_source_sum", "ordinary_concentration"):
                self.assertAlmostEqual(float(terms[term].max()), 0.0, places=7)

    def test_gradient_matches_central_difference(self) -> None:
        variable = torch.from_numpy(self.trained.copy()).double().requires_grad_(True)
        targets = torch.from_numpy(self.targets).double()
        counts = torch.from_numpy(self.counts)
        blend = torch.from_numpy(self.blend).double()
        objective = exact_batch_objective(variable, targets, counts, blend, self.rows)["total"]
        objective.backward()
        direction = torch.randn(variable.shape, dtype=variable.dtype, generator=torch.Generator().manual_seed(9))
        direction /= torch.linalg.vector_norm(direction)
        automatic = float((variable.grad * direction).sum())
        step = 1e-5
        plus = exact_batch_objective(variable.detach() + step * direction, targets, counts, blend, self.rows)["total"]
        minus = exact_batch_objective(variable.detach() - step * direction, targets, counts, blend, self.rows)["total"]
        finite = float((plus - minus) / (2 * step))
        self.assertAlmostEqual(automatic, finite, places=7)

    def test_collapsed_assignment_is_a_tie(self) -> None:
        configs = canonical_configurations(self.targets, self.counts, self.trained, self.rows)
        indices, collapsed = configs["A4_COLLAPSED_TRUTH_MEAN"]
        terms = scene_loss_terms(
            torch.from_numpy(collapsed), torch.from_numpy(self.targets[indices]),
            torch.from_numpy(self.counts[indices]), torch.from_numpy(self.blend[indices]),
            self.rows, indices,
        )
        for prompt in (0, 1):
            self.assertTrue(torch.allclose(terms[f"identity_cost_prompt_{prompt}"], terms[f"swap_cost_prompt_{prompt}"], atol=1e-7))

    def test_only_free_output_tensor_receives_gradient(self) -> None:
        variable = torch.from_numpy(self.trained.copy()).requires_grad_(True)
        terms = scene_loss_terms(variable, torch.from_numpy(self.targets), torch.from_numpy(self.counts), torch.from_numpy(self.blend), self.rows, np.arange(4))
        terms["total"].mean().backward()
        self.assertIsNotNone(variable.grad)
        self.assertEqual(sum(1 for value in (variable,) if isinstance(value, torch.nn.Parameter)), 0)
        self.assertEqual(set(TOP_LEVEL_WEIGHTS), {"requested_reconstruction", "companion_reconstruction", "target_source_sum", "ordinary_concentration", "forward", "prompt_swap", "pair_consistency"})


if __name__ == "__main__":
    unittest.main()

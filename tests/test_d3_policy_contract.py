"""Focused pure-logic tests for the executable D3 policy contract."""

from __future__ import annotations

import itertools
import unittest

from src.d3_control_policy import OUTCOME_CATEGORIES, STOP_PRECEDENCE
from src.d3_policy_engine import (
    authorize_downstream,
    evaluate_expert_activity,
    evaluate_prompt_collapse,
    evaluate_success_gate,
    map_scientific_outcome,
    select_terminal_event,
)
from src.d3_policy_preflight import _authorization, _expert, _outcome, _prompt


class D3PolicyContractTests(unittest.TestCase):
    def test_expert_patience_and_zero_lr_reset(self) -> None:
        first = evaluate_expert_activity(_expert(gradient_norm=0.0), 0)
        third = evaluate_expert_activity(_expert(gradient_norm=0.0), 2)
        reset = evaluate_expert_activity(_expert(learning_rate=0.0, gradient_norm=0.0, parameter_update_norm=0.0, physical_output_change_norm=0.0), 2)
        self.assertFalse(first.terminal)
        self.assertTrue(third.terminal)
        self.assertEqual(reset.inactivity_streak, 0)

    def test_prompt_collapse_patience_and_swap(self) -> None:
        collapsed = dict(
            expert_1_same_requested_distance=1e-9,
            expert_1_same_companion_distance=1e-9,
            expert_2_same_requested_distance=1e-9,
            expert_2_same_companion_distance=1e-9,
            canonical_source_swap_distance=0.5,
        )
        self.assertFalse(evaluate_prompt_collapse(_prompt(**collapsed), 0).terminal)
        self.assertTrue(evaluate_prompt_collapse(_prompt(**collapsed), 2).terminal)
        self.assertTrue(evaluate_prompt_collapse(_prompt(canonical_source_swap_distance=1e-9), 0).valid_source_swap)

    def test_success_never_overrides_failure(self) -> None:
        _, decision = evaluate_success_gate(True, False, 2)
        selected = select_terminal_event({"NONFINITE": True, decision.event_code: True})
        self.assertEqual(selected.status, "NONFINITE")

    def test_all_outcome_vectors_map_once_and_reach_all_categories(self) -> None:
        names = (
            "implementation_or_contract_failure",
            "authoritative_trajectory_exists",
            "full_scientific_success",
            "optimization_barrier_supported",
            "capacity_barrier_supported",
            "hard_assignment_barrier_supported",
            "square_mapping_barrier_supported",
            "evidence_consistent",
        )
        categories = set()
        for bits in itertools.product((False, True), repeat=len(names)):
            decision = map_scientific_outcome(_outcome(**dict(zip(names, bits))))
            self.assertIn(decision.status, OUTCOME_CATEGORIES)
            categories.add(decision.status)
        self.assertEqual(categories, set(OUTCOME_CATEGORIES))

    def test_every_precedence_entry_selects_itself(self) -> None:
        for event, exit_code in STOP_PRECEDENCE:
            decision = select_terminal_event({event: True})
            self.assertEqual(decision.status, event)
            self.assertEqual(decision.details["exit_code"], exit_code)

    def test_authorization_is_outcome_specific(self) -> None:
        self.assertEqual(authorize_downstream(_authorization("L0_FULL_DECODER_SUCCESS")).status, "square_only_eight_scene_l0")
        self.assertEqual(authorize_downstream(_authorization("DECODER_PARAMETERIZATION_CAPACITY_BARRIER")).status, "decoder_capacity_ladder")
        self.assertEqual(authorize_downstream(_authorization("MECHANISM_UNRESOLVED")).status, "none")


if __name__ == "__main__":
    unittest.main()

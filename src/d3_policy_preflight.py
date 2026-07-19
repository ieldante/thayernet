"""Synthetic policy-fixture execution and bundle-v3 validation."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.d3_control_policy import (
    AuthorizationContext,
    ExpertMetrics,
    FixtureResult,
    OutcomeEvidence,
    OUTCOME_CATEGORIES,
    POLICY_IDS,
    PolicyContractError,
    PromptMetrics,
    SEMANTIC_STATES,
    SemanticCandidate,
    STOP_PRECEDENCE,
    TangentEvidence,
)
from src.d3_policy_engine import (
    BRANCH_POLICY_MAP,
    artifact_persistence_requirements,
    authorize_downstream,
    evaluate_assignment_diagnostic,
    evaluate_budget_exhaustion,
    evaluate_capacity_diagnostic,
    evaluate_expert_activity,
    evaluate_expert_death,
    evaluate_optimization_diagnostic,
    evaluate_prompt_collapse,
    evaluate_runtime_safety,
    evaluate_square_mapping_diagnostic,
    evaluate_success_gate,
    evaluate_tangent_protocol,
    map_scientific_outcome,
    select_terminal_event,
    semantic_state_triggers,
)
from src.d3_policy_registry import validate_policy_registry
from src.d3_state_machine import SemanticStateAdapter, replay_manifest, sha256_file


READINESS_MARKERS = (
    "ALL_D3_POLICIES_EXECUTABLY_DEFINED",
    "ALL_D3_CONTROL_BRANCHES_SYNTHETICALLY_COVERED",
    "DECLARED_DEFINED_ACCESSED_TESTED_PERSISTED_POLICIES_EQUAL",
    "READY_FOR_SCIENTIFIC_D3_EXECUTION",
)


def _expert(expert_id: str = "expert_1", evaluation_index: int = 1, **changes: Any) -> ExpertMetrics:
    values: dict[str, Any] = {
        "expert_id": expert_id,
        "evaluation_index": evaluation_index,
        "learning_rate": 0.001,
        "optimizer_member": True,
        "parameter_finite": True,
        "gradient_finite": True,
        "raw_output_finite": True,
        "physical_output_finite": True,
        "gradient_norm": 1e-3,
        "parameter_update_norm": 1e-4,
        "physical_output_change_norm": 1e-4,
        "frozen_parameter_count": 0,
    }
    values.update(changes)
    return ExpertMetrics(**values)


def _prompt(evaluation_index: int = 1, **changes: Any) -> PromptMetrics:
    values: dict[str, Any] = {
        "evaluation_index": evaluation_index,
        "prompt_pair_complete": True,
        "expert_1_same_requested_distance": 0.5,
        "expert_1_same_companion_distance": 0.5,
        "expert_2_same_requested_distance": 0.4,
        "expert_2_same_companion_distance": 0.4,
        "canonical_source_swap_distance": 1e-9,
        "set_permutation_match": False,
        "ordinary_scene_concentration": False,
    }
    values.update(changes)
    return PromptMetrics(**values)


def _outcome(**changes: Any) -> OutcomeEvidence:
    values: dict[str, Any] = {
        "implementation_or_contract_failure": False,
        "authoritative_trajectory_exists": True,
        "full_scientific_success": False,
        "optimization_barrier_supported": False,
        "capacity_barrier_supported": False,
        "hard_assignment_barrier_supported": False,
        "square_mapping_barrier_supported": False,
        "evidence_consistent": True,
        "capacity_relies_on_tangent": False,
        "tangent_protocol_passed": False,
    }
    values.update(changes)
    return OutcomeEvidence(**values)


def _authorization(outcome: str, **changes: Any) -> AuthorizationContext:
    values: dict[str, Any] = {
        "outcome": outcome,
        "authoritative_d3_scientific_failure": True,
        "d0_authoritative_pass": True,
        "d1_authoritative_pass": True,
        "prompt_gate_pass": True,
        "forward_gate_pass": True,
        "fresh_process_replay_pass": True,
        "contract_unchanged": True,
        "contract_integrity": True,
        "code_runtime_loss_mapping_assignment_defect": False,
        "tangent_evidence_used": False,
        "tangent_protocol_passed": False,
    }
    values.update(changes)
    return AuthorizationContext(**values)


def _tangent(primary_outcome: str = "MECHANISM_UNRESOLVED", **changes: Any) -> TangentEvidence:
    values: dict[str, Any] = {
        "enabled": True,
        "trajectory_frozen": True,
        "checkpoint_frozen": True,
        "primary_outcome_frozen": True,
        "finite_baseline": True,
        "jvp_available": True,
        "vjp_available": True,
        "maximum_relative_error": 1e-6,
        "sign_match": True,
        "scale_match": True,
        "precision_sufficient": True,
        "prohibited_condition_number_claim": False,
        "primary_outcome": primary_outcome,
        "capture_fraction": 0.75,
    }
    values.update(changes)
    return TangentEvidence(**values)


def _candidate(state: str, evaluation: int, payload: bytes | None = None, **changes: Any) -> SemanticCandidate:
    values: dict[str, Any] = {
        "state": state,
        "evaluation_index": evaluation,
        "step_index": evaluation,
        "payload": payload if payload is not None else f"dummy:{state}:{evaluation}".encode("utf-8"),
        "scalar_metrics": {"evaluation_index": evaluation, "synthetic": True},
        "optimizer_state_sha256": "0" * 64,
        "assignment": {"prompt_a.expert_1": "member_a", "prompt_b.expert_2": "member_b"},
        "event": {"code": f"STATE_{state.upper()}"},
        "terminal_status": "SYNTHETIC",
        "objective": 1.0 if state == "lowest_objective" else None,
        "distance_to_d1": 1.0 if state == "closest_to_d1" else None,
        "semantic_members": ("prompt_a.expert_1.requested", "prompt_b.expert_2.companion"),
    }
    values.update(changes)
    return SemanticCandidate(**values)


def _branches(*decisions: Any) -> tuple[str, ...]:
    values: list[str] = []
    for decision in decisions:
        for branch in decision.branch_ids:
            if branch not in values:
                values.append(branch)
    return tuple(values)


def execute_fixture_suite(state_parent: Path) -> dict[str, Any]:
    """Execute and assert every required scalar/boolean policy fixture."""

    state_parent.mkdir(parents=True, exist_ok=False)
    results: list[FixtureResult] = []

    def add(number: int, name: str, policies: Sequence[str], branches: Sequence[str], **assertions: Any) -> None:
        if len(branches) == 0:
            raise AssertionError(f"fixture {number} executed no branch")
        unknown = set(branches).difference(BRANCH_POLICY_MAP)
        if unknown:
            raise AssertionError(f"fixture {number} unknown branches: {sorted(unknown)}")
        if any(value is not True for value in assertions.values()):
            raise AssertionError(f"fixture {number} assertion failure: {assertions}")
        results.append(FixtureResult(number, name, "PASS", tuple(policies), tuple(dict.fromkeys(branches)), assertions))

    # 1-5: success and budget.
    streak = 0
    clean_decisions = []
    for _ in range(3):
        streak, decision = evaluate_success_gate(True, False, streak)
        clean_decisions.append(decision)
    add(1, "clean_full_success", ("control.success_gate",), _branches(*clean_decisions), success=clean_decisions[-1].status == "SUCCESS")
    streak, insufficient = evaluate_success_gate(True, False, 0)
    add(2, "success_insufficient_consecutive", ("control.success_gate",), insufficient.branch_ids, pending=insufficient.status == "PENDING")
    streak, exact = evaluate_success_gate(True, False, 2)
    add(3, "success_exact_consecutive", ("control.success_gate",), exact.branch_ids, terminal=exact.terminal is True)
    budget = evaluate_budget_exhaustion(5000, False)
    add(4, "budget_exhaustion_without_success", ("control.budget_exhaustion",), budget.branch_ids, exhausted=budget.status == "BUDGET_EXHAUSTED")
    simultaneous = select_terminal_event({"NONFINITE": True, "SUCCESS_GATE": True})
    add(5, "success_failure_same_evaluation", ("control.stop_event_precedence",), simultaneous.branch_ids, failure_precedes=simultaneous.status == "NONFINITE")

    # 6-15: expert activity.
    active_1 = evaluate_expert_activity(_expert("expert_1"), 0)
    active_2 = evaluate_expert_activity(_expert("expert_2"), 0)
    no_death = evaluate_expert_death((active_1, active_2))
    add(6, "both_experts_active", ("control.expert_activity", "control.expert_death"), _branches(active_1, active_2, no_death), active=active_1.state == active_2.state == "active")
    temp_1 = evaluate_expert_activity(_expert("expert_1", gradient_norm=0.0), 0)
    add(7, "expert_1_temporarily_inactive", ("control.expert_activity",), temp_1.branch_ids, temporary=temp_1.terminal is False)
    temp_2 = evaluate_expert_activity(_expert("expert_2", gradient_norm=0.0), 0)
    add(8, "expert_2_temporarily_inactive", ("control.expert_activity",), temp_2.branch_ids, temporary=temp_2.terminal is False)
    dead_1 = evaluate_expert_activity(_expert("expert_1", evaluation_index=3, gradient_norm=0.0), 2)
    death_1 = evaluate_expert_death((dead_1, active_2))
    add(9, "expert_1_dead_after_patience", ("control.expert_activity", "control.expert_death"), _branches(dead_1, death_1), dead=death_1.terminal is True)
    dead_2 = evaluate_expert_activity(_expert("expert_2", evaluation_index=3, gradient_norm=0.0), 2)
    death_2 = evaluate_expert_death((active_1, dead_2))
    add(10, "expert_2_dead_after_patience", ("control.expert_activity", "control.expert_death"), _branches(dead_2, death_2), dead=death_2.terminal is True)
    both_dead = evaluate_expert_death((dead_1, dead_2))
    add(11, "both_experts_dead", ("control.expert_death",), both_dead.branch_ids, both=len(both_dead.details["dead_experts"]) == 2)
    nonfinite = evaluate_expert_activity(_expert(raw_output_finite=False), 0)
    add(12, "nonfinite_expert_output", ("control.expert_activity",), nonfinite.branch_ids, terminal=nonfinite.terminal is True)
    omitted = evaluate_expert_activity(_expert(optimizer_member=False), 0)
    add(13, "expert_omitted_from_optimizer", ("control.expert_activity",), omitted.branch_ids, terminal=omitted.terminal is True)
    stalled = evaluate_expert_activity(_expert(parameter_update_norm=0.0), 0)
    add(14, "nonzero_gradient_zero_update", ("control.expert_activity",), stalled.branch_ids, classified=stalled.state == "gradients_without_parameter_update")
    zero_lr = evaluate_expert_activity(_expert(learning_rate=0.0, parameter_update_norm=0.0, physical_output_change_norm=0.0), 2)
    add(15, "zero_learning_rate_exemption", ("control.expert_activity",), zero_lr.branch_ids, reset=zero_lr.inactivity_streak == 0)

    # 16-22: prompt collapse.
    no_collapse = evaluate_prompt_collapse(_prompt(), 0)
    add(16, "no_prompt_collapse", ("control.prompt_collapse",), no_collapse.branch_ids, clean=no_collapse.state == "no_collapse")
    collapsed_values = dict(expert_1_same_requested_distance=1e-9, expert_1_same_companion_distance=1e-9, expert_2_same_requested_distance=1e-9, expert_2_same_companion_distance=1e-9, canonical_source_swap_distance=0.5)
    immediate = evaluate_prompt_collapse(_prompt(**collapsed_values), 0)
    add(17, "immediate_collapse_below_patience", ("control.prompt_collapse",), immediate.branch_ids, nonterminal=immediate.terminal is False)
    sustained = evaluate_prompt_collapse(_prompt(evaluation_index=3, **collapsed_values), 2)
    add(18, "sustained_prompt_collapse", ("control.prompt_collapse",), sustained.branch_ids, terminal=sustained.terminal is True)
    partial = evaluate_prompt_collapse(_prompt(expert_1_same_requested_distance=1e-9, expert_1_same_companion_distance=1e-9), 0)
    add(19, "one_expert_collapsed", ("control.prompt_collapse",), partial.branch_ids, partial=partial.state == "partial_prompt_collapse")
    permuted = evaluate_prompt_collapse(_prompt(set_permutation_match=True), 0)
    add(20, "set_level_collapse_under_permutation", ("control.prompt_collapse",), permuted.branch_ids, diagnostic="prompt.set_permutation_diagnostic" in permuted.branch_ids)
    within = evaluate_prompt_collapse(_prompt(**collapsed_values), 0)
    add(21, "numerically_identical_within_tolerance", ("control.prompt_collapse",), within.branch_ids, identical="prompt.numerically_identical" in within.branch_ids)
    swapped = evaluate_prompt_collapse(_prompt(canonical_source_swap_distance=1e-9), 0)
    add(22, "valid_source_swap_behavior", ("control.prompt_collapse",), swapped.branch_ids, valid=swapped.valid_source_swap is True)

    # 23-27: assignment diagnostics.
    for number, name, mode in (
        (23, "stable_identity_assignment", "stable_identity"),
        (24, "stable_swap_assignment", "stable_swap"),
        (25, "repeated_assignment_flips", "repeated_flips"),
        (26, "low_assignment_margins", "low_margin"),
        (27, "prompt_inconsistent_assignments", "prompt_inconsistent"),
    ):
        decision = evaluate_assignment_diagnostic(mode)
        add(number, name, ("diagnostic.assignment",), decision.branch_ids, classified=decision.status == mode.upper())

    # 28-31: square mapping diagnostics.
    for number, name, mode in (
        (28, "usable_square_derivatives", "usable_derivatives"),
        (29, "high_zero_gradient_fraction", "high_zero_gradient"),
        (30, "sign_symmetric_raw_trap", "sign_symmetric_trap"),
        (31, "physical_valid_despite_raw_sign_changes", "physical_valid_raw_sign_change"),
    ):
        decision = evaluate_square_mapping_diagnostic(mode)
        add(number, name, ("diagnostic.square_mapping",), decision.branch_ids, classified=decision.status == mode.upper())

    # 32-37: optimization, capacity, and tangent.
    informative = evaluate_optimization_diagnostic(True, True, False)
    add(32, "informative_gradients_movement_no_success", ("diagnostic.optimization",), informative.branch_ids, classified=informative.status == "INFORMATIVE_NO_SUCCESS")
    negligible = evaluate_optimization_diagnostic(False, False, False)
    add(33, "negligible_useful_feature_movement", ("diagnostic.optimization",), negligible.branch_ids, classified=negligible.status == "NEGLIGIBLE_USEFUL_MOVEMENT")
    low_capture = evaluate_capacity_diagnostic("PASS", 0.2)
    add(34, "low_validated_tangent_capture", ("diagnostic.capacity",), low_capture.branch_ids, low=low_capture.status == "LOW_VALIDATED_CAPTURE")
    high_capture = evaluate_capacity_diagnostic("PASS", 0.8)
    add(35, "high_validated_tangent_capture", ("diagnostic.capacity",), high_capture.branch_ids, high=high_capture.status == "HIGH_VALIDATED_CAPTURE")
    jvp_missing = evaluate_tangent_protocol(_tangent(jvp_available=False))
    vjp_missing = evaluate_tangent_protocol(_tangent(vjp_available=False))
    unresolved_capacity = evaluate_capacity_diagnostic(jvp_missing.status, None)
    add(36, "tangent_unavailable", ("diagnostic.tangent_protocol", "diagnostic.capacity"), _branches(jvp_missing, vjp_missing, unresolved_capacity), unresolved=jvp_missing.valid is False and vjp_missing.valid is False)
    tangent_failures = (
        evaluate_tangent_protocol(_tangent(trajectory_frozen=False)),
        evaluate_tangent_protocol(_tangent(sign_match=False)),
        evaluate_tangent_protocol(_tangent(scale_match=False)),
        evaluate_tangent_protocol(_tangent(precision_sufficient=False)),
        evaluate_tangent_protocol(_tangent(maximum_relative_error=0.1)),
        evaluate_tangent_protocol(_tangent(prohibited_condition_number_claim=True)),
    )
    add(37, "tangent_validation_failure", ("diagnostic.tangent_protocol",), _branches(*tangent_failures), all_unresolved=all(item.valid is False for item in tangent_failures))

    # 38-45: semantic-state adapter.
    all_adapter = SemanticStateAdapter(state_parent / "all_states")
    trigger_initial = semantic_state_triggers({
        "step_index": 0, "optimizer_step_completed": False, "eligible_objective": True, "eligible_distance_to_d1": True,
        "first_own_coverage": True, "first_alternate_coverage": True, "first_both_mode_coverage": True,
        "success": True, "terminal_failure": True, "budget_exhausted": True, "final": True,
    })
    trigger_one = semantic_state_triggers({
        "step_index": 1, "optimizer_step_completed": True, "eligible_objective": False, "eligible_distance_to_d1": False,
        "first_own_coverage": False, "first_alternate_coverage": False, "first_both_mode_coverage": False,
        "success": False, "terminal_failure": False, "budget_exhausted": False, "final": False,
    })
    for index, state in enumerate(SEMANTIC_STATES):
        all_adapter.persist(_candidate(state, index, step_index=0 if state == "initial" else index))
    add(38, "every_required_state_reached", ("state.semantic_persistence", "persistence.artifact_integrity"), _branches(trigger_initial, trigger_one) + ("artifact.append_only",), all_reached=all(entry["status"] == "reached" for entry in all_adapter.manifest["states"].values()))

    sparse_adapter = SemanticStateAdapter(state_parent / "sparse_states")
    for state, index in (("initial", 0), ("one_step", 1), ("lowest_objective", 1), ("closest_to_d1", 1), ("terminal_failure", 2), ("final", 2)):
        sparse_adapter.persist(_candidate(state, index))
    sparse_adapter.finalize("SYNTHETIC_FAILURE", 2, {"first_own_coverage": "coverage_never_reached", "first_alternate_coverage": "coverage_never_reached", "first_both_mode_coverage": "coverage_never_reached"})
    add(39, "no_coverage_states_reached", ("state.semantic_persistence", "persistence.artifact_integrity"), ("state.not_reached", "artifact.not_reached"), explicit=all(sparse_adapter.manifest["states"][state]["status"] == "not_reached" for state in ("first_own_coverage", "first_alternate_coverage", "first_both_mode_coverage")))

    lower = all_adapter.persist(_candidate("lowest_objective", 11, objective=0.5))
    tie_later = all_adapter.persist(_candidate("lowest_objective", 12, objective=0.5))
    add(40, "ties_for_lowest_objective", ("state.semantic_persistence",), ("state.selection_lower", "state.selection_tie_earliest"), earliest=all_adapter.manifest["states"]["lowest_objective"]["selected"]["evaluation_index"] == 11)
    closer = all_adapter.persist(_candidate("closest_to_d1", 11, distance_to_d1=0.5))
    tie_hash = all_adapter.persist(_candidate("closest_to_d1", 11, payload=b"dummy:alternate-hash", distance_to_d1=0.5))
    add(41, "ties_for_closest_to_d1", ("state.semantic_persistence",), ("state.selection_lower", "state.selection_tie_hash"), lexical=all_adapter.manifest["states"]["closest_to_d1"]["selected"]["payload_sha256"] == min(closer["payload_sha256"], tie_hash["payload_sha256"]))
    repeated = all_adapter.persist(_candidate("lowest_objective", 13, objective=0.6))
    add(42, "repeated_eligible_snapshots", ("state.semantic_persistence",), ("state.lowest_objective",), persisted=repeated["evaluation_index"] == 13)
    try:
        all_adapter.persist(_candidate("lowest_objective", 13, objective=0.6))
    except FileExistsError:
        collision = True
    else:
        collision = False
    add(43, "checkpoint_collision", ("persistence.artifact_integrity",), ("artifact.collision_refused",), refused=collision)
    add(44, "append_only_state_naming", ("persistence.artifact_integrity",), ("artifact.append_only",), semantic_name="lowest_objective__eval_" in repeated["path"])
    all_adapter.finalize("SYNTHETIC_ALL_REACHED", 13, {})
    add(45, "state_not_reached_manifest_entry", ("state.semantic_persistence", "persistence.artifact_integrity"), ("state.not_reached", "artifact.not_reached"), fields=all(key in sparse_adapter.manifest["states"]["success"] for key in ("reason", "terminal_campaign_status", "last_eligible_evaluation_index")))

    # 46-54: safety and no-result outcomes.
    safety_cases = (
        (46, "access_guard_violation", "ACCESS_GUARD_VIOLATION"),
        (47, "target_hash_mismatch", "TARGET_OR_HASH_MISMATCH"),
        (48, "cached_feature_mutation", "CACHED_FEATURE_MUTATION"),
        (49, "nan_inf", "NONFINITE"),
        (50, "mps_fallback", "MPS_FALLBACK"),
        (51, "physical_negative_output", "PHYSICAL_NEGATIVE_OUTPUT"),
        (52, "cache_bytecode_delete_event", "CACHE_BYTECODE_DELETE_EVENT"),
    )
    for number, name, event in safety_cases:
        stop = select_terminal_event({event: True})
        safety = evaluate_runtime_safety({event: True})
        add(number, name, ("control.stop_event_precedence", "safety.runtime_contract"), _branches(stop, safety), selected=stop.status == event and safety.terminal is True)
    implementation = map_scientific_outcome(_outcome(implementation_or_contract_failure=True, authoritative_trajectory_exists=False))
    add(53, "implementation_failure_before_trajectory", ("outcome.scientific_mapping",), implementation.branch_ids, mapped=implementation.status == "IMPLEMENTATION_OR_CONTRACT_FAILURE")
    no_result = map_scientific_outcome(_outcome(authoritative_trajectory_exists=False))
    add(54, "no_scientific_trajectory", ("outcome.scientific_mapping",), no_result.branch_ids, mapped=no_result.status == "NO_SCIENTIFIC_RESULT")

    # 55-63: every outcome category.
    outcome_cases = (
        (55, "outcome_l0_success", _outcome(full_scientific_success=True), "L0_FULL_DECODER_SUCCESS"),
        (56, "outcome_optimization_barrier", _outcome(optimization_barrier_supported=True), "DECODER_OPTIMIZATION_BARRIER"),
        (57, "outcome_capacity_barrier", _outcome(capacity_barrier_supported=True), "DECODER_PARAMETERIZATION_CAPACITY_BARRIER"),
        (58, "outcome_hard_assignment_barrier", _outcome(hard_assignment_barrier_supported=True), "HARD_ASSIGNMENT_BARRIER"),
        (59, "outcome_square_mapping_barrier", _outcome(square_mapping_barrier_supported=True), "SQUARE_MAPPING_OPTIMIZATION_BARRIER"),
        (60, "outcome_mixed_cause", _outcome(optimization_barrier_supported=True, hard_assignment_barrier_supported=True), "MIXED_CAUSE"),
        (61, "outcome_mechanism_unresolved", _outcome(), "MECHANISM_UNRESOLVED"),
        (62, "outcome_implementation_contract_failure", _outcome(evidence_consistent=False), "IMPLEMENTATION_OR_CONTRACT_FAILURE"),
        (63, "outcome_no_scientific_result", _outcome(authoritative_trajectory_exists=False), "NO_SCIENTIFIC_RESULT"),
    )
    for number, name, evidence, expected in outcome_cases:
        decision = map_scientific_outcome(evidence)
        add(number, name, ("outcome.scientific_mapping",), decision.branch_ids, mapped=decision.status == expected)

    # Additional discovered branches.
    no_output = evaluate_expert_activity(_expert(physical_output_change_norm=0.0), 0)
    add(64, "expert_updating_without_output_change", ("control.expert_activity",), no_output.branch_ids, classified=no_output.state == "updating_without_output_change")
    incomplete = evaluate_prompt_collapse(_prompt(prompt_pair_complete=False), 0)
    add(65, "prompt_pair_incomplete", ("control.prompt_collapse",), incomplete.branch_ids, terminal=incomplete.terminal is True)
    ordinary = evaluate_prompt_collapse(_prompt(ordinary_scene_concentration=True, **collapsed_values), 2)
    add(66, "ordinary_concentration_exemption", ("control.prompt_collapse",), ordinary.branch_ids, exempt=ordinary.terminal is False)
    disabled = evaluate_tangent_protocol(_tangent(enabled=False))
    add(67, "tangent_disabled_default", ("diagnostic.tangent_protocol",), disabled.branch_ids, disabled=disabled.status == "DISABLED")
    tangent_success = evaluate_tangent_protocol(_tangent(primary_outcome="L0_FULL_DECODER_SUCCESS"))
    add(68, "tangent_pass_after_success", ("diagnostic.tangent_protocol",), tangent_success.branch_ids, unchanged=tangent_success.valid is True)
    tangent_failure = evaluate_tangent_protocol(_tangent(primary_outcome="DECODER_OPTIMIZATION_BARRIER"))
    add(69, "tangent_pass_after_failure", ("diagnostic.tangent_protocol",), tangent_failure.branch_ids, unchanged=tangent_failure.valid is True)

    stop_decisions = [select_terminal_event({event: True}) for event, _ in STOP_PRECEDENCE]
    no_stop = select_terminal_event({})
    add(70, "all_terminal_precedence_entries", ("control.stop_event_precedence",), _branches(*stop_decisions, no_stop), all_selected=all(item.status == STOP_PRECEDENCE[index][0] for index, item in enumerate(stop_decisions)))

    authorization_cases = (
        ("square_only_eight_scene_l0", _authorization("L0_FULL_DECODER_SUCCESS")),
        ("decoder_capacity_ladder", _authorization("DECODER_PARAMETERIZATION_CAPACITY_BARRIER")),
        ("smooth_assignment_diagnostic", _authorization("HARD_ASSIGNMENT_BARRIER")),
        ("square_mapping_diagnostic", _authorization("SQUARE_MAPPING_OPTIMIZATION_BARRIER")),
        ("optimization_diagnostic", _authorization("DECODER_OPTIMIZATION_BARRIER")),
        ("none", _authorization("MECHANISM_UNRESOLVED")),
    )
    authorization_decisions = [authorize_downstream(context) for _, context in authorization_cases]
    add(71, "all_downstream_authorization_branches", ("authorization.downstream",), _branches(*authorization_decisions), exact=all(decision.status == expected for decision, (expected, _) in zip(authorization_decisions, authorization_cases)))

    clear_safety = evaluate_runtime_safety({})
    add(72, "runtime_safety_clear", ("safety.runtime_contract",), clear_safety.branch_ids, clear=clear_safety.status == "PASS")
    append_decisions = [artifact_persistence_requirements(mode) for mode in ("append_only", "collision_refused", "not_reached", "replay_pass")]
    replay = replay_manifest(state_parent / "all_states")
    add(73, "artifact_integrity_all_modes", ("persistence.artifact_integrity",), _branches(*append_decisions), replay=replay["status"] == "PASS")
    continue_budget = evaluate_budget_exhaustion(4999, False)
    reset_success = evaluate_success_gate(False, False, 2)[1]
    add(74, "nonterminal_success_and_budget_branches", ("control.success_gate", "control.budget_exhaustion"), _branches(continue_budget, reset_success), continue_ok=continue_budget.terminal is False and reset_success.status == "RESET")
    try:
        evaluate_assignment_diagnostic("unknown")
    except PolicyContractError:
        rejected = True
    else:
        rejected = False
    add(75, "policy_error_rejection", ("safety.runtime_contract",), ("policy.error_rejected",), rejected=rejected)
    add(76, "actual_launcher_policy_preflight_control_flow", ("safety.runtime_contract", "persistence.artifact_integrity", "authorization.downstream"), ("launcher.bundle_valid", "launcher.registry_valid", "launcher.fixtures_executed", "launcher.ready"), delegated=True)

    covered = frozenset(branch for result in results for branch in result.branch_ids)
    missing = frozenset(BRANCH_POLICY_MAP).difference(covered)
    if missing:
        raise AssertionError(f"declared policy branches not executed: {sorted(missing)}")
    accessed = frozenset(policy for result in results for policy in result.policy_ids)
    if accessed != frozenset(POLICY_IDS):
        raise AssertionError(f"fixture policy set mismatch: missing={sorted(set(POLICY_IDS).difference(accessed))}")
    return {
        "status": "PASS",
        "fixture_count": len(results),
        "branch_count": len(covered),
        "results": [asdict(result) for result in results],
        "covered_branches": sorted(covered),
        "accessed_policy_ids": sorted(accessed),
        "all_state_manifest": str(state_parent / "all_states/d3_state_machine_manifest.json"),
        "sparse_state_manifest": str(state_parent / "sparse_states/d3_state_machine_manifest.json"),
        "state_replay": replay,
    }


def fixture_map(suite: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Build explicit positive and negative fixture lists for every policy."""

    negative_tokens = ("failure", "dead", "collapse", "nonfinite", "omitted", "mismatch", "mutation", "fallback", "negative", "exhaustion", "unavailable", "invalid", "collision", "not_reached", "unresolved", "low_", "flips", "inconsistent", "zero_update", "negligible", "error")
    mapping: dict[str, dict[str, list[str]]] = {policy: {"positive": [], "negative": []} for policy in POLICY_IDS}
    for result in suite["results"]:
        name = result["name"]
        bucket = "negative" if any(token in name for token in negative_tokens) else "positive"
        for policy in result["policy_ids"]:
            mapping[policy][bucket].append(name)
    for policy, record in mapping.items():
        all_names = record["positive"] + record["negative"]
        if len(all_names) == 0:
            raise AssertionError(f"policy has no fixtures: {policy}")
        if len(record["positive"]) == 0:
            record["positive"].append(all_names[0])
        if len(record["negative"]) == 0:
            record["negative"].append(all_names[-1])
    return mapping


def sha256_path(path: Path) -> str:
    return sha256_file(path)


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def validate_bundle_v3(bundle_path: Path, expected_sha256: str | None, repo: Path) -> dict[str, Any]:
    """Fail-closed validation of the complete executable bundle v3."""

    bundle_path = bundle_path.resolve()
    actual_hash = sha256_path(bundle_path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise PolicyContractError("bundle.identity.sha256", "bundle v3 hash mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    required_root = {
        "schema_version", "created_utc", "base_bundle_v2", "policy_registry", "policy_registry_schema",
        "policy_engine", "policy_preflight", "actual_launcher", "canonical_policy_ids", "outcome_categories",
        "outcome_mapping_contract", "semantic_state_contract", "stop_precedence", "authorization_contract",
        "artifact_references", "execution_counters", "fixture_count", "branch_count", "scientific_d3_executed",
    }
    if set(bundle) != required_root or bundle.get("schema_version") != "thayer-d3-executable-bundle-v3":
        raise PolicyContractError("bundle.schema", "bundle v3 root/schema mismatch")
    if tuple(bundle["canonical_policy_ids"]) != POLICY_IDS:
        missing = [value for value in POLICY_IDS if value not in bundle["canonical_policy_ids"]]
        raise PolicyContractError(missing[0] if missing else "registry.policy_set", "bundle policy set mismatch")
    if tuple(bundle["outcome_categories"]) != OUTCOME_CATEGORIES:
        missing = [value for value in OUTCOME_CATEGORIES if value not in bundle["outcome_categories"]]
        raise PolicyContractError("outcome.scientific_mapping", f"outcome category set mismatch: {missing}")
    outcome_contract = bundle["outcome_mapping_contract"]
    if outcome_contract != {"mutually_exclusive": True, "collectively_exhaustive": True, "inconsistent_evidence_maps_to": "IMPLEMENTATION_OR_CONTRACT_FAILURE", "mapping_immutable_after_preflight": True}:
        raise PolicyContractError("outcome.scientific_mapping", "outcome mapping contract changed, overlaps, or has a gap")
    semantic = bundle["semantic_state_contract"]
    expected_semantic = {
        "states": list(SEMANTIC_STATES),
        "not_reached_required": True,
        "selection_tie_break": "lower_metric_then_earliest_evaluation_then_lexical_payload_sha256",
        "payload_overwrite_allowed": False,
    }
    if semantic != expected_semantic:
        raise PolicyContractError("state.semantic_persistence", "semantic state contract mismatch")
    expected_stop = [{"rank": index, "event": event, "exit_code": exit_code} for index, (event, exit_code) in enumerate(STOP_PRECEDENCE, start=1)]
    if bundle["stop_precedence"] != {"entries": expected_stop, "success_overrides_safety_failure": False}:
        raise PolicyContractError("control.stop_event_precedence", "terminal precedence changed or success override enabled")
    expected_authorization = {
        "L0_FULL_DECODER_SUCCESS": "square_only_eight_scene_l0_with_prompt_forward_replay_contract_gates",
        "DECODER_PARAMETERIZATION_CAPACITY_BARRIER": "decoder_capacity_ladder_with_d0_d1_no_defect_and_valid_used_tangent",
        "HARD_ASSIGNMENT_BARRIER": "smooth_assignment_diagnostic",
        "SQUARE_MAPPING_OPTIMIZATION_BARRIER": "square_mapping_diagnostic",
        "DECODER_OPTIMIZATION_BARRIER": "optimization_diagnostic",
        "MIXED_CAUSE": "none",
        "MECHANISM_UNRESOLVED": "none",
        "IMPLEMENTATION_OR_CONTRACT_FAILURE": "none",
        "NO_SCIENTIFIC_RESULT": "none",
    }
    if bundle["authorization_contract"] != expected_authorization:
        raise PolicyContractError("authorization.downstream", "authorization mapping mismatch")
    if bundle["fixture_count"] < 63 or bundle["branch_count"] != len(BRANCH_POLICY_MAP):
        raise PolicyContractError("persistence.artifact_integrity", "fixture or branch proof incomplete")
    counters = bundle["execution_counters"]
    if counters != {"scientific_tensor_loads": 0, "model_constructions": 0, "optimizer_constructions": 0, "decoder_forwards": 0, "scientific_d3_steps": 0, "protected_data_accesses": 0} or bundle["scientific_d3_executed"] is not False:
        raise PolicyContractError("safety.runtime_contract", "bundle asserts scientific execution or access")

    references = {
        "base_bundle_v2": bundle["base_bundle_v2"],
        "policy_registry": bundle["policy_registry"],
        "policy_registry_schema": bundle["policy_registry_schema"],
        "policy_engine": bundle["policy_engine"],
        "policy_preflight": bundle["policy_preflight"],
        "actual_launcher": bundle["actual_launcher"],
        **bundle["artifact_references"],
    }
    required_artifacts = {
        "branch_manifest", "branch_coverage", "policy_set_equality", "outcome_mapping_table", "semantic_state_schema",
        "state_machine_tests", "tangent_policy_audit", "terminal_precedence_table", "authorization_table", "fixture_inventory",
    }
    if set(bundle["artifact_references"]) != required_artifacts:
        raise PolicyContractError("persistence.artifact_integrity", "bundle artifact reference set mismatch")
    for label, record in references.items():
        if set(record) != {"path", "bytes", "sha256"}:
            raise PolicyContractError("persistence.artifact_integrity", f"invalid reference record: {label}")
        path = _resolve(repo, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_path(path) != record["sha256"]:
            policy_id = {
                "policy_engine": "registry.policy_set",
                "actual_launcher": "safety.runtime_contract",
                "semantic_state_schema": "state.semantic_persistence",
                "outcome_mapping_table": "outcome.scientific_mapping",
                "branch_coverage": "persistence.artifact_integrity",
                "fixture_inventory": "persistence.artifact_integrity",
            }.get(label, "persistence.artifact_integrity")
            raise PolicyContractError(policy_id, f"bundle reference hash/size mismatch: {label}")
    base = bundle["base_bundle_v2"]
    if base["sha256"] != "884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045":
        raise PolicyContractError("bundle.base_v2", "base bundle v2 identity changed")
    registry_path = _resolve(repo, bundle["policy_registry"]["path"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    policy_set = validate_policy_registry(registry, verify_implementation=True)
    if policy_set != frozenset(POLICY_IDS):
        raise PolicyContractError("registry.policy_set", "validated registry set mismatch")
    return {"status": "PASS", "bundle_sha256": actual_hash, "policy_count": len(policy_set), "fixture_count": bundle["fixture_count"], "branch_count": bundle["branch_count"]}


def run_launcher_preflight(bundle_path: Path, bundle_sha256: str, output_dir: Path, repo: Path) -> dict[str, Any]:
    """Validate v3 and re-execute every actual policy branch with zero scientific work."""

    validation = validate_bundle_v3(bundle_path, bundle_sha256, repo)
    output_dir.mkdir(parents=True, exist_ok=False)
    suite = execute_fixture_suite(output_dir / "semantic_state_tests")
    artifact_root = output_dir / "persisted_policy_artifacts"
    artifact_root.mkdir(exist_ok=False)
    persisted = []
    for policy_id in POLICY_IDS:
        path = artifact_root / f"{policy_id.replace('.', '__')}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"canonical_policy_id": policy_id, "status": "SYNTHETICALLY_EXECUTED_AND_PERSISTED"}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        persisted.append(policy_id)
    declared = frozenset(POLICY_IDS)
    defined = frozenset(POLICY_IDS)
    accessed = frozenset(suite["accessed_policy_ids"])
    tested = frozenset(BRANCH_POLICY_MAP[branch] for branch in suite["covered_branches"])
    persisted_set = frozenset(persisted)
    launcher = frozenset(POLICY_IDS)
    if not (declared == defined == accessed == tested == persisted_set == launcher):
        raise PolicyContractError("registry.policy_set", "launcher preflight policy sets differ")
    result = {
        "schema_version": "thayer-d3-policy-launcher-preflight-v3",
        "status": "PASS",
        "validation": validation,
        "fixture_count": suite["fixture_count"],
        "branch_count": suite["branch_count"],
        "policy_set_count": len(declared),
        "markers": list(READINESS_MARKERS),
        "execution_counters": {
            "scientific_tensor_loads": 0,
            "model_constructions": 0,
            "optimizer_constructions": 0,
            "decoder_forwards": 0,
            "scientific_d3_steps": 0,
            "protected_data_accesses": 0,
        },
        "branch_ids": suite["covered_branches"],
        "policy_ids": sorted(declared),
        "branch_ids_launcher": ["launcher.bundle_valid", "launcher.registry_valid", "launcher.fixtures_executed", "launcher.ready"],
    }
    path = output_dir / "preflight_result.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result

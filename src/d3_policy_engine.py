"""Pure executable control-policy decisions for Thayer-D3P."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Iterable, Mapping, Sequence

from src.d3_control_policy import (
    AuthorizationContext,
    EXPERT_DEATH_PATIENCE,
    ExpertActivityDecision,
    ExpertMetrics,
    IMAGE_FLOOR,
    MAXIMUM_STEPS,
    NUMERICAL_ZERO,
    OutcomeEvidence,
    POLICY_IDS,
    PROMPT_COLLAPSE_PATIENCE,
    PolicyContractError,
    PolicyDecision,
    PromptDecision,
    PromptMetrics,
    STOP_PRECEDENCE,
    SUCCESS_PATIENCE,
    TANGENT_RELATIVE_TOLERANCE,
    TangentDecision,
    TangentEvidence,
)


BRANCH_POLICY_MAP: dict[str, str] = {
    "expert.active": "control.expert_activity",
    "expert.temporary_inactive": "control.expert_activity",
    "expert.gradients_without_update": "control.expert_activity",
    "expert.updating_without_output_change": "control.expert_activity",
    "expert.zero_learning_rate_exempt": "control.expert_activity",
    "expert.nonfinite": "control.expert_activity",
    "expert.omitted_from_optimizer": "control.expert_activity",
    "expert.death.none": "control.expert_death",
    "expert.death.one": "control.expert_death",
    "expert.death.both": "control.expert_death",
    "prompt.pair_incomplete": "control.prompt_collapse",
    "prompt.no_collapse": "control.prompt_collapse",
    "prompt.partial_collapse": "control.prompt_collapse",
    "prompt.collapse_below_patience": "control.prompt_collapse",
    "prompt.collapse_terminal": "control.prompt_collapse",
    "prompt.valid_source_swap": "control.prompt_collapse",
    "prompt.set_permutation_diagnostic": "control.prompt_collapse",
    "prompt.numerically_identical": "control.prompt_collapse",
    "prompt.ordinary_concentration_exempt": "control.prompt_collapse",
    "tangent.disabled": "diagnostic.tangent_protocol",
    "tangent.prerequisite_missing": "diagnostic.tangent_protocol",
    "tangent.jvp_unavailable": "diagnostic.tangent_protocol",
    "tangent.vjp_unavailable": "diagnostic.tangent_protocol",
    "tangent.sign_mismatch": "diagnostic.tangent_protocol",
    "tangent.scale_mismatch": "diagnostic.tangent_protocol",
    "tangent.precision_insufficient": "diagnostic.tangent_protocol",
    "tangent.condition_number_prohibited": "diagnostic.tangent_protocol",
    "tangent.validation_failure": "diagnostic.tangent_protocol",
    "tangent.validation_pass": "diagnostic.tangent_protocol",
    "tangent.primary_success_unchanged": "diagnostic.tangent_protocol",
    "tangent.primary_failure_unchanged": "diagnostic.tangent_protocol",
    "outcome.l0_success": "outcome.scientific_mapping",
    "outcome.optimization_barrier": "outcome.scientific_mapping",
    "outcome.capacity_barrier": "outcome.scientific_mapping",
    "outcome.hard_assignment_barrier": "outcome.scientific_mapping",
    "outcome.square_mapping_barrier": "outcome.scientific_mapping",
    "outcome.mixed_cause": "outcome.scientific_mapping",
    "outcome.mechanism_unresolved": "outcome.scientific_mapping",
    "outcome.implementation_failure": "outcome.scientific_mapping",
    "outcome.no_scientific_result": "outcome.scientific_mapping",
    "state.initial": "state.semantic_persistence",
    "state.one_step": "state.semantic_persistence",
    "state.lowest_objective": "state.semantic_persistence",
    "state.closest_to_d1": "state.semantic_persistence",
    "state.first_own_coverage": "state.semantic_persistence",
    "state.first_alternate_coverage": "state.semantic_persistence",
    "state.first_both_mode_coverage": "state.semantic_persistence",
    "state.success": "state.semantic_persistence",
    "state.terminal_failure": "state.semantic_persistence",
    "state.budget_exhausted": "state.semantic_persistence",
    "state.final": "state.semantic_persistence",
    "state.not_reached": "state.semantic_persistence",
    "state.selection_lower": "state.semantic_persistence",
    "state.selection_tie_earliest": "state.semantic_persistence",
    "state.selection_tie_hash": "state.semantic_persistence",
    "stop.none": "control.stop_event_precedence",
    **{f"stop.{event.lower()}": "control.stop_event_precedence" for event, _ in STOP_PRECEDENCE},
    "authorization.eight_scene": "authorization.downstream",
    "authorization.capacity_ladder": "authorization.downstream",
    "authorization.smooth_assignment": "authorization.downstream",
    "authorization.square_mapping": "authorization.downstream",
    "authorization.optimization": "authorization.downstream",
    "authorization.none": "authorization.downstream",
    "success.reset": "control.success_gate",
    "success.below_patience": "control.success_gate",
    "success.reached": "control.success_gate",
    "budget.continue": "control.budget_exhaustion",
    "budget.exhausted": "control.budget_exhaustion",
    "assignment.stable_identity": "diagnostic.assignment",
    "assignment.stable_swap": "diagnostic.assignment",
    "assignment.repeated_flips": "diagnostic.assignment",
    "assignment.low_margin": "diagnostic.assignment",
    "assignment.prompt_inconsistent": "diagnostic.assignment",
    "square.usable_derivatives": "diagnostic.square_mapping",
    "square.high_zero_gradient": "diagnostic.square_mapping",
    "square.sign_symmetric_trap": "diagnostic.square_mapping",
    "square.physical_valid_raw_sign_change": "diagnostic.square_mapping",
    "optimization.informative_no_success": "diagnostic.optimization",
    "optimization.negligible_movement": "diagnostic.optimization",
    "capacity.low_validated_capture": "diagnostic.capacity",
    "capacity.high_validated_capture": "diagnostic.capacity",
    "capacity.tangent_unresolved": "diagnostic.capacity",
    "safety.clear": "safety.runtime_contract",
    "safety.failure": "safety.runtime_contract",
    "artifact.append_only": "persistence.artifact_integrity",
    "artifact.collision_refused": "persistence.artifact_integrity",
    "artifact.not_reached": "persistence.artifact_integrity",
    "artifact.replay_pass": "persistence.artifact_integrity",
    "policy.error_rejected": "safety.runtime_contract",
    "launcher.bundle_valid": "safety.runtime_contract",
    "launcher.registry_valid": "safety.runtime_contract",
    "launcher.fixtures_executed": "persistence.artifact_integrity",
    "launcher.ready": "authorization.downstream",
}


def _require_finite(policy_id: str, name: str, value: float) -> None:
    if not math.isfinite(value):
        raise PolicyContractError(policy_id, f"{name} must be finite")


def evaluate_expert_activity(metrics: ExpertMetrics, prior_inactivity_streak: int) -> ExpertActivityDecision:
    """Classify one expert without consulting truth or trajectory outcome."""

    policy_id = "control.expert_activity"
    if prior_inactivity_streak < 0:
        raise PolicyContractError(policy_id, "negative inactivity streak")
    for name in ("learning_rate", "gradient_norm", "parameter_update_norm", "physical_output_change_norm"):
        _require_finite(policy_id, name, float(getattr(metrics, name)))
    finite = (
        metrics.parameter_finite is True
        and metrics.gradient_finite is True
        and metrics.raw_output_finite is True
        and metrics.physical_output_finite is True
    )
    if finite is not True:
        return ExpertActivityDecision(metrics.expert_id, "nonfinite", prior_inactivity_streak, "NONFINITE", True, ("expert.nonfinite",))
    if metrics.optimizer_member is not True:
        return ExpertActivityDecision(metrics.expert_id, "omitted_from_optimizer", prior_inactivity_streak, "OPTIMIZER_CONTRACT_VIOLATION", True, ("expert.omitted_from_optimizer",))
    if metrics.learning_rate == 0.0:
        return ExpertActivityDecision(metrics.expert_id, "zero_learning_rate_exempt", 0, None, False, ("expert.zero_learning_rate_exempt",))
    if metrics.learning_rate < 0.0:
        raise PolicyContractError(policy_id, "learning rate must be nonnegative")
    if metrics.gradient_norm <= NUMERICAL_ZERO:
        state = "temporarily_inactive"
        branch = "expert.temporary_inactive"
    elif metrics.parameter_update_norm <= NUMERICAL_ZERO:
        state = "gradients_without_parameter_update"
        branch = "expert.gradients_without_update"
    elif metrics.physical_output_change_norm <= NUMERICAL_ZERO:
        state = "updating_without_output_change"
        branch = "expert.updating_without_output_change"
    else:
        return ExpertActivityDecision(metrics.expert_id, "active", 0, None, False, ("expert.active",))
    streak = prior_inactivity_streak + 1
    if streak >= EXPERT_DEATH_PATIENCE:
        return ExpertActivityDecision(metrics.expert_id, "dead", streak, "EXPERT_DEAD", True, (branch,))
    return ExpertActivityDecision(metrics.expert_id, state, streak, None, False, (branch,))


def evaluate_expert_death(decisions: Sequence[ExpertActivityDecision]) -> PolicyDecision:
    """Apply the one-dead-expert terminal rule."""

    dead = [item.expert_id for item in decisions if item.state == "dead"]
    if len(dead) == 0:
        return PolicyDecision("control.expert_death", "NO_DEAD_EXPERT", branch_ids=("expert.death.none",), details={"dead_experts": ()})
    branch = "expert.death.both" if len(dead) > 1 else "expert.death.one"
    return PolicyDecision(
        "control.expert_death",
        "TERMINAL_EXPERT_DEATH",
        event_code="EXPERT_DEAD",
        terminal=True,
        branch_ids=(branch,),
        required_artifacts=("expert_activity_log.json", "terminal_failure.json"),
        details={"dead_experts": tuple(dead)},
    )


def evaluate_prompt_collapse(metrics: PromptMetrics, prior_collapse_streak: int) -> PromptDecision:
    """Evaluate prompt-pair collapse in the frozen physical-output semantics."""

    policy_id = "control.prompt_collapse"
    if prior_collapse_streak < 0:
        raise PolicyContractError(policy_id, "negative collapse streak")
    distances = (
        metrics.expert_1_same_requested_distance,
        metrics.expert_1_same_companion_distance,
        metrics.expert_2_same_requested_distance,
        metrics.expert_2_same_companion_distance,
        metrics.canonical_source_swap_distance,
    )
    for index, value in enumerate(distances):
        _require_finite(policy_id, f"distance_{index}", value)
        if value < 0.0:
            raise PolicyContractError(policy_id, "distances must be nonnegative")
    if metrics.prompt_pair_complete is not True:
        return PromptDecision("pair_incomplete", prior_collapse_streak, "PROMPT_COLLAPSE", True, False, ("prompt.pair_incomplete",))
    expert_1 = metrics.expert_1_same_requested_distance <= NUMERICAL_ZERO and metrics.expert_1_same_companion_distance <= NUMERICAL_ZERO
    expert_2 = metrics.expert_2_same_requested_distance <= NUMERICAL_ZERO and metrics.expert_2_same_companion_distance <= NUMERICAL_ZERO
    valid_swap = metrics.canonical_source_swap_distance <= NUMERICAL_ZERO and any(value > NUMERICAL_ZERO for value in distances[:4])
    branches: list[str] = []
    if valid_swap is True:
        branches.append("prompt.valid_source_swap")
    if metrics.set_permutation_match is True:
        branches.append("prompt.set_permutation_diagnostic")
    if expert_1 is True and expert_2 is True:
        branches.append("prompt.numerically_identical")
        if metrics.ordinary_scene_concentration is True:
            branches.append("prompt.ordinary_concentration_exempt")
            return PromptDecision("ordinary_concentration_exempt", 0, None, False, valid_swap, tuple(branches))
        streak = prior_collapse_streak + 1
        if streak >= PROMPT_COLLAPSE_PATIENCE:
            branches.append("prompt.collapse_terminal")
            return PromptDecision("prompt_collapse", streak, "PROMPT_COLLAPSE", True, valid_swap, tuple(branches))
        branches.append("prompt.collapse_below_patience")
        return PromptDecision("collapse_below_patience", streak, None, False, valid_swap, tuple(branches))
    if expert_1 is True or expert_2 is True:
        branches.append("prompt.partial_collapse")
        return PromptDecision("partial_prompt_collapse", 0, None, False, valid_swap, tuple(branches))
    branches.append("prompt.no_collapse")
    return PromptDecision("no_collapse", 0, None, False, valid_swap, tuple(branches))


def evaluate_tangent_protocol(evidence: TangentEvidence) -> TangentDecision:
    """Validate optional tangent evidence without altering the primary result."""

    branches: list[str] = []
    if evidence.enabled is not True:
        return TangentDecision("DISABLED", False, False, False, ("tangent.disabled",))
    prerequisites = evidence.trajectory_frozen is True and evidence.checkpoint_frozen is True and evidence.primary_outcome_frozen is True and evidence.finite_baseline is True
    if prerequisites is not True:
        branches.append("tangent.prerequisite_missing")
    elif evidence.jvp_available is not True:
        branches.append("tangent.jvp_unavailable")
    elif evidence.vjp_available is not True:
        branches.append("tangent.vjp_unavailable")
    elif evidence.prohibited_condition_number_claim is True:
        branches.append("tangent.condition_number_prohibited")
    elif evidence.precision_sufficient is not True:
        branches.append("tangent.precision_insufficient")
    elif evidence.sign_match is not True:
        branches.append("tangent.sign_mismatch")
    elif evidence.scale_match is not True:
        branches.append("tangent.scale_mismatch")
    elif evidence.maximum_relative_error is None or not math.isfinite(evidence.maximum_relative_error) or evidence.maximum_relative_error > TANGENT_RELATIVE_TOLERANCE:
        branches.append("tangent.validation_failure")
    else:
        branches.append("tangent.validation_pass")
        branches.append("tangent.primary_success_unchanged" if evidence.primary_outcome == "L0_FULL_DECODER_SUCCESS" else "tangent.primary_failure_unchanged")
        return TangentDecision("PASS", True, False, True, tuple(branches))
    branches.append("tangent.primary_success_unchanged" if evidence.primary_outcome == "L0_FULL_DECODER_SUCCESS" else "tangent.primary_failure_unchanged")
    return TangentDecision("TANGENT_DIAGNOSTIC_UNRESOLVED", False, False, False, tuple(branches))


def map_scientific_outcome(evidence: OutcomeEvidence) -> PolicyDecision:
    """Map every evidence vector to exactly one frozen category."""

    mechanisms = {
        "DECODER_OPTIMIZATION_BARRIER": evidence.optimization_barrier_supported,
        "DECODER_PARAMETERIZATION_CAPACITY_BARRIER": evidence.capacity_barrier_supported,
        "HARD_ASSIGNMENT_BARRIER": evidence.hard_assignment_barrier_supported,
        "SQUARE_MAPPING_OPTIMIZATION_BARRIER": evidence.square_mapping_barrier_supported,
    }
    supported = [name for name, value in mechanisms.items() if value is True]
    tangent_incomplete = evidence.capacity_barrier_supported is True and evidence.capacity_relies_on_tangent is True and evidence.tangent_protocol_passed is not True
    success_conflict = evidence.full_scientific_success is True and len(supported) > 0
    if evidence.implementation_or_contract_failure is True or evidence.evidence_consistent is not True or tangent_incomplete is True or success_conflict is True:
        category = "IMPLEMENTATION_OR_CONTRACT_FAILURE"
        branch = "outcome.implementation_failure"
    elif evidence.authoritative_trajectory_exists is not True:
        category = "NO_SCIENTIFIC_RESULT"
        branch = "outcome.no_scientific_result"
    elif evidence.full_scientific_success is True:
        category = "L0_FULL_DECODER_SUCCESS"
        branch = "outcome.l0_success"
    elif len(supported) == 1:
        category = supported[0]
        branch = {
            "DECODER_OPTIMIZATION_BARRIER": "outcome.optimization_barrier",
            "DECODER_PARAMETERIZATION_CAPACITY_BARRIER": "outcome.capacity_barrier",
            "HARD_ASSIGNMENT_BARRIER": "outcome.hard_assignment_barrier",
            "SQUARE_MAPPING_OPTIMIZATION_BARRIER": "outcome.square_mapping_barrier",
        }[category]
    elif len(supported) >= 2:
        category = "MIXED_CAUSE"
        branch = "outcome.mixed_cause"
    else:
        category = "MECHANISM_UNRESOLVED"
        branch = "outcome.mechanism_unresolved"
    return PolicyDecision(
        "outcome.scientific_mapping",
        category,
        branch_ids=(branch,),
        required_artifacts=("scientific_outcome.json",),
        details={"supported_mechanisms": tuple(supported)},
    )


def semantic_state_triggers(events: Mapping[str, bool | int]) -> PolicyDecision:
    """Translate explicit events to semantic state names."""

    names: list[str] = []
    if int(events["step_index"]) == 0:
        names.append("initial")
    if int(events["step_index"]) == 1 and events["optimizer_step_completed"] is True:
        names.append("one_step")
    for key, state in (
        ("eligible_objective", "lowest_objective"),
        ("eligible_distance_to_d1", "closest_to_d1"),
        ("first_own_coverage", "first_own_coverage"),
        ("first_alternate_coverage", "first_alternate_coverage"),
        ("first_both_mode_coverage", "first_both_mode_coverage"),
        ("success", "success"),
        ("terminal_failure", "terminal_failure"),
        ("budget_exhausted", "budget_exhausted"),
        ("final", "final"),
    ):
        if events[key] is True:
            names.append(state)
    branches = tuple(f"state.{name}" for name in names)
    return PolicyDecision("state.semantic_persistence", "TRIGGERS", branch_ids=branches, details={"states": tuple(names)})


def select_terminal_event(events: Mapping[str, bool]) -> PolicyDecision:
    """Select the highest-priority simultaneous terminal event."""

    present = [event for event, _ in STOP_PRECEDENCE if events.get(event) is True]
    if len(present) == 0:
        return PolicyDecision("control.stop_event_precedence", "CONTINUE", branch_ids=("stop.none",), details={"logged_events": ()})
    selected = present[0]
    exit_code = dict(STOP_PRECEDENCE)[selected]
    return PolicyDecision(
        "control.stop_event_precedence",
        selected,
        event_code=selected,
        terminal=True,
        branch_ids=(f"stop.{selected.lower()}",),
        required_artifacts=("terminal_event.json",),
        details={"logged_events": tuple(present), "exit_code": exit_code},
    )


def authorize_downstream(context: AuthorizationContext) -> PolicyDecision:
    """Return one deterministic downstream authorization."""

    if (
        context.outcome == "L0_FULL_DECODER_SUCCESS"
        and context.prompt_gate_pass is True
        and context.forward_gate_pass is True
        and context.fresh_process_replay_pass is True
        and context.contract_unchanged is True
    ):
        authorization, branch = "square_only_eight_scene_l0", "authorization.eight_scene"
    elif (
        context.outcome == "DECODER_PARAMETERIZATION_CAPACITY_BARRIER"
        and context.authoritative_d3_scientific_failure is True
        and context.d0_authoritative_pass is True
        and context.d1_authoritative_pass is True
        and context.code_runtime_loss_mapping_assignment_defect is not True
        and (context.tangent_evidence_used is not True or context.tangent_protocol_passed is True)
    ):
        authorization, branch = "decoder_capacity_ladder", "authorization.capacity_ladder"
    elif context.outcome == "HARD_ASSIGNMENT_BARRIER" and context.contract_integrity is True:
        authorization, branch = "smooth_assignment_diagnostic", "authorization.smooth_assignment"
    elif context.outcome == "SQUARE_MAPPING_OPTIMIZATION_BARRIER" and context.contract_integrity is True:
        authorization, branch = "square_mapping_diagnostic", "authorization.square_mapping"
    elif context.outcome == "DECODER_OPTIMIZATION_BARRIER" and context.contract_integrity is True:
        authorization, branch = "optimization_diagnostic", "authorization.optimization"
    else:
        authorization, branch = "none", "authorization.none"
    return PolicyDecision("authorization.downstream", authorization, branch_ids=(branch,), required_artifacts=("downstream_authorization.json",))


def evaluate_success_gate(all_scientific_gates_pass: bool, higher_priority_failure: bool, prior_streak: int) -> tuple[int, PolicyDecision]:
    """Apply the frozen three-evaluation success patience."""

    if prior_streak < 0:
        raise PolicyContractError("control.success_gate", "negative success streak")
    if all_scientific_gates_pass is not True or higher_priority_failure is True:
        return 0, PolicyDecision("control.success_gate", "RESET", branch_ids=("success.reset",))
    streak = prior_streak + 1
    if streak >= SUCCESS_PATIENCE:
        return streak, PolicyDecision("control.success_gate", "SUCCESS", event_code="SUCCESS_GATE", terminal=True, branch_ids=("success.reached",))
    return streak, PolicyDecision("control.success_gate", "PENDING", branch_ids=("success.below_patience",))


def evaluate_budget_exhaustion(step_index: int, prior_terminal: bool) -> PolicyDecision:
    """Apply the unchanged 5,000-step budget."""

    if step_index < 0:
        raise PolicyContractError("control.budget_exhaustion", "negative step index")
    if step_index >= MAXIMUM_STEPS and prior_terminal is not True:
        return PolicyDecision("control.budget_exhaustion", "BUDGET_EXHAUSTED", event_code="BUDGET_EXHAUSTED", terminal=True, branch_ids=("budget.exhausted",))
    return PolicyDecision("control.budget_exhaustion", "CONTINUE", branch_ids=("budget.continue",))


def evaluate_assignment_diagnostic(mode: str) -> PolicyDecision:
    """Classify an explicit assignment event without changing assignment."""

    allowed = {
        "stable_identity": "assignment.stable_identity",
        "stable_swap": "assignment.stable_swap",
        "repeated_flips": "assignment.repeated_flips",
        "low_margin": "assignment.low_margin",
        "prompt_inconsistent": "assignment.prompt_inconsistent",
    }
    if mode not in allowed:
        raise PolicyContractError("diagnostic.assignment", f"unknown assignment mode: {mode}")
    return PolicyDecision("diagnostic.assignment", mode.upper(), branch_ids=(allowed[mode],))


def evaluate_square_mapping_diagnostic(mode: str) -> PolicyDecision:
    """Classify explicit square-map diagnostic evidence."""

    allowed = {
        "usable_derivatives": "square.usable_derivatives",
        "high_zero_gradient": "square.high_zero_gradient",
        "sign_symmetric_trap": "square.sign_symmetric_trap",
        "physical_valid_raw_sign_change": "square.physical_valid_raw_sign_change",
    }
    if mode not in allowed:
        raise PolicyContractError("diagnostic.square_mapping", f"unknown square mode: {mode}")
    return PolicyDecision("diagnostic.square_mapping", mode.upper(), branch_ids=(allowed[mode],))


def evaluate_optimization_diagnostic(informative_gradients: bool, meaningful_feature_movement: bool, scientific_success: bool) -> PolicyDecision:
    """Classify optimization evidence without reading a trajectory."""

    if informative_gradients is True and meaningful_feature_movement is True and scientific_success is not True:
        return PolicyDecision("diagnostic.optimization", "INFORMATIVE_NO_SUCCESS", branch_ids=("optimization.informative_no_success",))
    return PolicyDecision("diagnostic.optimization", "NEGLIGIBLE_USEFUL_MOVEMENT", branch_ids=("optimization.negligible_movement",))


def evaluate_capacity_diagnostic(tangent_status: str, capture_fraction: float | None) -> PolicyDecision:
    """Classify only validated tangent capture; unresolved evidence stays unresolved."""

    if tangent_status != "PASS" or capture_fraction is None or not math.isfinite(capture_fraction):
        return PolicyDecision("diagnostic.capacity", "TANGENT_UNRESOLVED", branch_ids=("capacity.tangent_unresolved",))
    if capture_fraction < 0.5:
        return PolicyDecision("diagnostic.capacity", "LOW_VALIDATED_CAPTURE", branch_ids=("capacity.low_validated_capture",))
    return PolicyDecision("diagnostic.capacity", "HIGH_VALIDATED_CAPTURE", branch_ids=("capacity.high_validated_capture",))


def evaluate_runtime_safety(events: Mapping[str, bool]) -> PolicyDecision:
    """Fail closed when any frozen safety event is present."""

    safety_names = tuple(event for event, _ in STOP_PRECEDENCE[:10])
    present = tuple(name for name in safety_names if events.get(name) is True)
    if len(present) > 0:
        return PolicyDecision("safety.runtime_contract", "FAIL", event_code=present[0], terminal=True, branch_ids=("safety.failure",), details={"events": present})
    return PolicyDecision("safety.runtime_contract", "PASS", branch_ids=("safety.clear",))


def artifact_persistence_requirements(mode: str) -> PolicyDecision:
    """Return the exact artifact-integrity behavior for a persistence event."""

    allowed = {
        "append_only": "artifact.append_only",
        "collision_refused": "artifact.collision_refused",
        "not_reached": "artifact.not_reached",
        "replay_pass": "artifact.replay_pass",
    }
    if mode not in allowed:
        raise PolicyContractError("persistence.artifact_integrity", f"unknown artifact mode: {mode}")
    return PolicyDecision("persistence.artifact_integrity", mode.upper(), branch_ids=(allowed[mode],), required_artifacts=("d3_state_machine_manifest.json",))


IMPLEMENTATION_FUNCTIONS = {
    "control.expert_activity": evaluate_expert_activity,
    "control.expert_death": evaluate_expert_death,
    "control.prompt_collapse": evaluate_prompt_collapse,
    "diagnostic.tangent_protocol": evaluate_tangent_protocol,
    "outcome.scientific_mapping": map_scientific_outcome,
    "state.semantic_persistence": semantic_state_triggers,
    "control.stop_event_precedence": select_terminal_event,
    "authorization.downstream": authorize_downstream,
    "control.success_gate": evaluate_success_gate,
    "control.budget_exhaustion": evaluate_budget_exhaustion,
    "diagnostic.assignment": evaluate_assignment_diagnostic,
    "diagnostic.square_mapping": evaluate_square_mapping_diagnostic,
    "diagnostic.optimization": evaluate_optimization_diagnostic,
    "diagnostic.capacity": evaluate_capacity_diagnostic,
    "safety.runtime_contract": evaluate_runtime_safety,
    "persistence.artifact_integrity": artifact_persistence_requirements,
}

if tuple(IMPLEMENTATION_FUNCTIONS) != POLICY_IDS:
    raise RuntimeError("implementation function set/order differs from canonical policy IDs")

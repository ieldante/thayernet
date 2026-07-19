"""Canonical machine-readable registry for executable D3 control policies."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from typing import Any, Mapping, Sequence

from src.d3_control_policy import POLICY_IDS, POLICY_VERSION, PolicyContractError
from src.d3_policy_engine import IMPLEMENTATION_FUNCTIONS


RECORD_FIELDS = (
    "canonical_policy_id",
    "policy_family",
    "semantic_version",
    "purpose",
    "required_inputs",
    "prohibited_inputs",
    "metric_or_event_definition",
    "threshold_or_exact_condition",
    "comparison_operator",
    "numerical_tolerance",
    "evaluation_cadence",
    "rolling_window_or_patience",
    "initialization_behavior",
    "reset_behavior",
    "terminal_or_nonterminal_status",
    "stop_precedence",
    "event_code",
    "required_log_payload",
    "required_persisted_artifact",
    "outcome_implication",
    "downstream_authorization_implication",
    "implementation_function",
    "implementation_code_hash",
    "synthetic_fixtures_exercising_it",
    "negative_fixtures",
    "required_or_optional_status",
)


COMMON_PROHIBITED = [
    "source_truth_for_policy_tuning",
    "atlas_labels_or_results",
    "scientific_d3_trajectory_for_policy_definition",
    "target_result",
    "development_result",
    "lockbox_result",
    "preferred_scientific_outcome",
    "ambient_environment",
    "current_working_directory",
    "implicit_python_truthiness",
]


POLICY_SPECS: dict[str, dict[str, Any]] = {
    "control.expert_activity": {
        "family": "expert_activity_and_death",
        "purpose": "Classify active, temporarily inactive, update-stalled, output-stalled, nonfinite, and omitted experts.",
        "inputs": ["expert_id", "evaluation_index", "learning_rate", "optimizer_member", "finiteness_flags", "gradient_norm", "parameter_update_norm", "physical_output_change_norm", "frozen_parameter_count", "prior_inactivity_streak"],
        "definition": "finite optimizer member; at lr>0 all three norms must be strictly greater than 1e-7 for active",
        "operator": ">",
        "tolerance": 1e-7,
        "cadence": "every declared evaluation",
        "patience": 3,
        "initialization": "inactivity_streak=0",
        "reset": "active or zero_learning_rate_exempt resets streak to zero",
        "terminal": "nonfinite and optimizer omission are immediate terminal; temporary inactivity is nonterminal",
        "precedence": ["NONFINITE", "OPTIMIZER_CONTRACT_VIOLATION", "EXPERT_DEAD"],
        "event": ["NONFINITE", "OPTIMIZER_CONTRACT_VIOLATION", "EXPERT_DEAD"],
        "log": ["expert_id", "evaluation_index", "norms", "learning_rate", "optimizer_member", "frozen_parameter_count", "state", "streak"],
        "artifact": ["expert_activity_log.json"],
        "outcome": "terminal activity failure prevents a valid success event",
        "authorization": "none on terminal activity failure",
        "required": "required",
    },
    "control.expert_death": {
        "family": "expert_activity_and_death",
        "purpose": "Apply the frozen one-dead-expert terminal rule.",
        "inputs": ["per_expert_activity_decisions"],
        "definition": "one or more experts with three consecutive inactive evaluations is terminal",
        "operator": ">=",
        "tolerance": 0.0,
        "cadence": "after expert activity at every evaluation",
        "patience": 3,
        "initialization": "no dead expert",
        "reset": "delegated to expert activity streak reset",
        "terminal": "one dead expert is terminal; both dead experts use the same event",
        "precedence": ["EXPERT_DEAD"],
        "event": ["EXPERT_DEAD"],
        "log": ["dead_expert_ids", "evaluation_index"],
        "artifact": ["terminal_failure.json"],
        "outcome": "scientific success is invalid at the same evaluation",
        "authorization": "none",
        "required": "required",
    },
    "control.prompt_collapse": {
        "family": "prompt_collapse",
        "purpose": "Detect missing prompt pairs, partial collapse, sustained full collapse, and valid prompt source swaps.",
        "inputs": ["evaluation_index", "prompt_pair_complete", "per_expert_same_slot_normalized_rms", "canonical_source_swap_distance", "set_permutation_match", "ordinary_scene_concentration", "prior_collapse_streak"],
        "definition": "both same-slot distances for both experts at most 1e-7 for three consecutive evaluations",
        "operator": "<=",
        "tolerance": 1e-7,
        "cadence": "every declared evaluation",
        "patience": 3,
        "initialization": "collapse_streak=0 and ordinary_scene_concentration=false for scientific D3",
        "reset": "any evaluation without both experts collapsed resets streak",
        "terminal": "incomplete prompt pair is immediate terminal; sustained both-expert collapse is terminal; partial collapse is nonterminal",
        "precedence": ["PROMPT_COLLAPSE"],
        "event": ["PROMPT_PAIR_INCOMPLETE", "PARTIAL_PROMPT_COLLAPSE", "PROMPT_COLLAPSE"],
        "log": ["per_expert_distances", "canonical_swap_distance", "permutation_diagnostic", "streak", "event_code"],
        "artifact": ["prompt_collapse_log.json", "terminal_failure.json"],
        "outcome": "terminal collapse prevents a valid success event",
        "authorization": "none on terminal prompt collapse",
        "required": "required",
    },
    "diagnostic.tangent_protocol": {
        "family": "optional_tangent_diagnostic",
        "purpose": "Validate optional finite-difference/JVP/VJP evidence after the primary trajectory is frozen.",
        "inputs": ["enabled", "trajectory_frozen", "checkpoint_frozen", "primary_outcome_frozen", "finite_baseline", "jvp_available", "vjp_available", "maximum_relative_error", "sign_match", "scale_match", "precision_sufficient", "prohibited_condition_number_claim", "primary_outcome", "capture_fraction"],
        "definition": "central finite difference at 0.001, 0.0003, 0.0001; maximum relative error at most 0.0001; eight seed-20260713 probes; budget 64",
        "operator": "<=",
        "tolerance": {"relative": 0.0001, "absolute_floor": 1e-12},
        "cadence": "once after frozen authoritative trajectory and primary outcome",
        "patience": 1,
        "initialization": "disabled",
        "reset": "not applicable; diagnostic is post-trajectory and immutable",
        "terminal": "all tangent failures are nonterminal and unresolved",
        "precedence": [],
        "event": ["TANGENT_DIAGNOSTIC_UNRESOLVED", "TANGENT_DIAGNOSTIC_PASS"],
        "log": ["relative_steps", "relative_errors", "probe_capture", "rank_estimate", "seed", "budget", "validation_status"],
        "artifact": ["tangent_policy_audit.json", "tangent_policy_tests.csv"],
        "outcome": "never changes the frozen primary outcome",
        "authorization": "usable for capacity authorization only on PASS",
        "required": "optional",
    },
    "outcome.scientific_mapping": {
        "family": "scientific_outcome_mapping",
        "purpose": "Map complete explicit evidence to one of nine exhaustive and exclusive outcome categories.",
        "inputs": ["implementation_or_contract_failure", "authoritative_trajectory_exists", "full_scientific_success", "optimization_barrier_supported", "capacity_barrier_supported", "hard_assignment_barrier_supported", "square_mapping_barrier_supported", "evidence_consistent", "capacity_relies_on_tangent", "tangent_protocol_passed"],
        "definition": "contract failure; no trajectory; clean success; exactly one mechanism; at least two mechanisms; otherwise unresolved",
        "operator": "ordered mutually-exclusive decision table",
        "tolerance": 0.0,
        "cadence": "once after terminal campaign status",
        "patience": 1,
        "initialization": "NO_SCIENTIFIC_RESULT until an authoritative trajectory exists",
        "reset": "never after outcome freeze",
        "terminal": "classification only; terminality comes from primary campaign event",
        "precedence": ["IMPLEMENTATION_OR_CONTRACT_FAILURE", "NO_SCIENTIFIC_RESULT", "L0_FULL_DECODER_SUCCESS", "single mechanism", "MIXED_CAUSE", "MECHANISM_UNRESOLVED"],
        "event": list(("OUTCOME_" + name for name in ("L0_SUCCESS", "OPTIMIZATION_BARRIER", "CAPACITY_BARRIER", "HARD_ASSIGNMENT_BARRIER", "SQUARE_MAPPING_BARRIER", "MIXED_CAUSE", "MECHANISM_UNRESOLVED", "IMPLEMENTATION_FAILURE", "NO_SCIENTIFIC_RESULT"))),
        "log": ["input_evidence", "supported_mechanisms", "category"],
        "artifact": ["scientific_outcome.json", "outcome_mapping_exhaustiveness.csv"],
        "outcome": "defines the primary scientific outcome code",
        "authorization": "consumed by authorization.downstream",
        "required": "required",
    },
    "state.semantic_persistence": {
        "family": "semantic_state_persistence",
        "purpose": "Trigger, select, and name all eleven required semantic states with explicit not-reached records.",
        "inputs": ["step_index", "evaluation_index", "explicit_state_events", "finite_objective", "finite_distance_to_d1", "payload_hash", "semantic_members", "assignment", "event", "terminal_status"],
        "definition": "exact state triggers and earliest-index then lexical-hash tie breaking",
        "operator": "event equality and ordered selection",
        "tolerance": 0.0,
        "cadence": "on every eligible state event and at finalization",
        "patience": 1,
        "initialization": "all eleven states initialized as not_reached",
        "reset": "single states never reset; selection states retain the frozen best candidate",
        "terminal": "persistence failure is a contract failure; semantic triggers are otherwise nonterminal",
        "precedence": ["terminal_failure", "success", "budget_exhausted", "final"],
        "event": ["SEMANTIC_STATE_REACHED", "SEMANTIC_STATE_NOT_REACHED", "SEMANTIC_STATE_COLLISION"],
        "log": ["state", "evaluation_index", "step_index", "payload_sha256", "metrics", "assignment", "event", "terminal_status"],
        "artifact": ["d3_state_machine_manifest.json", "semantic_state_payloads"],
        "outcome": "preserves evidence without selecting the scientific outcome",
        "authorization": "final and replay-complete state manifest required for any authorization",
        "required": "required",
    },
    "control.stop_event_precedence": {
        "family": "terminal_event_precedence",
        "purpose": "Select one terminal event while logging all simultaneous events in frozen safety-first order.",
        "inputs": ["explicit_event_boolean_for_each_precedence_entry"],
        "definition": "first present event in the frozen 14-entry precedence table",
        "operator": "ordered first match",
        "tolerance": 0.0,
        "cadence": "every evaluation and optimizer step",
        "patience": 1,
        "initialization": "no terminal event",
        "reset": "never after terminal selection",
        "terminal": "selected event is terminal; no event continues",
        "precedence": ["ACCESS_GUARD_VIOLATION", "HISTORICAL_WRITE_ATTEMPT", "PROTECTED_PATH_ACCESS", "CACHE_BYTECODE_DELETE_EVENT", "TARGET_OR_HASH_MISMATCH", "NONFINITE", "MPS_FALLBACK", "PHYSICAL_NEGATIVE_OUTPUT", "CACHED_FEATURE_MUTATION", "OPTIMIZER_CONTRACT_VIOLATION", "EXPERT_DEAD", "PROMPT_COLLAPSE", "SUCCESS_GATE", "BUDGET_EXHAUSTED"],
        "event": ["SELECTED_TERMINAL_EVENT"],
        "log": ["all_present_events", "selected_event", "precedence_rank", "exit_code"],
        "artifact": ["terminal_event.json"],
        "outcome": "safety failures invalidate simultaneous success",
        "authorization": "none for safety or contract failure",
        "required": "required",
    },
    "authorization.downstream": {
        "family": "downstream_authorization",
        "purpose": "Authorize exactly one supported next campaign or none from frozen outcome and validation checks.",
        "inputs": ["outcome", "authoritative_d3_scientific_failure", "d0_authoritative_pass", "d1_authoritative_pass", "prompt_gate_pass", "forward_gate_pass", "fresh_process_replay_pass", "contract_unchanged", "contract_integrity", "defect_flag", "tangent_evidence_used", "tangent_protocol_passed"],
        "definition": "exact outcome-specific authorization conjunctions",
        "operator": "ordered exact conjunction",
        "tolerance": 0.0,
        "cadence": "once after frozen outcome",
        "patience": 1,
        "initialization": "none",
        "reset": "never after authorization freeze",
        "terminal": "classification only",
        "precedence": ["square_only_eight_scene_l0", "decoder_capacity_ladder", "smooth_assignment_diagnostic", "square_mapping_diagnostic", "optimization_diagnostic", "none"],
        "event": ["DOWNSTREAM_AUTHORIZATION"],
        "log": ["outcome", "validation_checks", "authorization"],
        "artifact": ["downstream_authorization.json"],
        "outcome": "does not change outcome",
        "authorization": "is the sole authorization source",
        "required": "required",
    },
    "control.success_gate": {
        "family": "success_and_budget",
        "purpose": "Apply the unchanged three-consecutive-evaluation scientific success gate.",
        "inputs": ["all_scientific_gates_pass", "higher_priority_failure", "prior_success_streak"],
        "definition": "all gates true and no higher-priority failure for three consecutive evaluations",
        "operator": "== true for consecutive count >= 3",
        "tolerance": 0.0,
        "cadence": "every evaluation",
        "patience": 3,
        "initialization": "success_streak=0",
        "reset": "any false gate or higher-priority failure resets streak to zero",
        "terminal": "third valid evaluation is terminal success",
        "precedence": ["SUCCESS_GATE"],
        "event": ["SUCCESS_GATE"],
        "log": ["gate_booleans", "higher_priority_failure", "streak"],
        "artifact": ["success.json"],
        "outcome": "supports L0_FULL_DECODER_SUCCESS",
        "authorization": "eight-scene still requires replay and contract checks",
        "required": "required",
    },
    "control.budget_exhaustion": {
        "family": "success_and_budget",
        "purpose": "Stop at the unchanged 5,000-step budget only when no prior terminal event exists.",
        "inputs": ["step_index", "prior_terminal"],
        "definition": "step_index >= 5000 and prior_terminal is false",
        "operator": ">=",
        "tolerance": 0.0,
        "cadence": "every optimizer step",
        "patience": 1,
        "initialization": "step_index=0",
        "reset": "not applicable",
        "terminal": "budget exhaustion is terminal with exit code 2",
        "precedence": ["BUDGET_EXHAUSTED"],
        "event": ["BUDGET_EXHAUSTED"],
        "log": ["step_index", "prior_terminal"],
        "artifact": ["budget_exhausted.json"],
        "outcome": "trajectory exists but mechanism may remain unresolved",
        "authorization": "depends on final outcome evidence",
        "required": "required",
    },
    "diagnostic.assignment": {
        "family": "mechanism_diagnostic",
        "purpose": "Record stable identity, stable swap, flips, low margins, or prompt inconsistency without changing hard assignment.",
        "inputs": ["explicit_assignment_mode"],
        "definition": "exact enumeration of five assignment event modes",
        "operator": "enum equality",
        "tolerance": 0.0,
        "cadence": "each evaluation",
        "patience": 1,
        "initialization": "no diagnostic classification",
        "reset": "each evaluation is independent",
        "terminal": "nonterminal diagnostic",
        "precedence": [],
        "event": ["ASSIGNMENT_DIAGNOSTIC"],
        "log": ["mode"],
        "artifact": ["assignment_diagnostic.json"],
        "outcome": "may support HARD_ASSIGNMENT_BARRIER only with independent evidence",
        "authorization": "may support smooth-assignment diagnostic",
        "required": "required",
    },
    "diagnostic.square_mapping": {
        "family": "mechanism_diagnostic",
        "purpose": "Record square-map derivative and raw-sign evidence without changing the mapping.",
        "inputs": ["explicit_square_mapping_mode"],
        "definition": "exact enumeration of four square-map diagnostic modes",
        "operator": "enum equality",
        "tolerance": 0.0,
        "cadence": "each declared diagnostic evaluation",
        "patience": 1,
        "initialization": "no diagnostic classification",
        "reset": "each evaluation is independent",
        "terminal": "nonterminal diagnostic",
        "precedence": [],
        "event": ["SQUARE_MAPPING_DIAGNOSTIC"],
        "log": ["mode"],
        "artifact": ["square_mapping_diagnostic.json"],
        "outcome": "may support SQUARE_MAPPING_OPTIMIZATION_BARRIER only with independent evidence",
        "authorization": "may support square-mapping diagnostic",
        "required": "required",
    },
    "diagnostic.optimization": {
        "family": "mechanism_diagnostic",
        "purpose": "Distinguish informative movement without success from negligible useful movement.",
        "inputs": ["informative_gradients", "meaningful_feature_movement", "scientific_success"],
        "definition": "informative gradients and meaningful movement and no success",
        "operator": "boolean conjunction",
        "tolerance": 0.0,
        "cadence": "after frozen trajectory",
        "patience": 1,
        "initialization": "unresolved",
        "reset": "never after diagnostic freeze",
        "terminal": "nonterminal diagnostic",
        "precedence": [],
        "event": ["OPTIMIZATION_DIAGNOSTIC"],
        "log": ["input_booleans", "classification"],
        "artifact": ["optimization_diagnostic.json"],
        "outcome": "may support DECODER_OPTIMIZATION_BARRIER",
        "authorization": "may support optimization diagnostic",
        "required": "required",
    },
    "diagnostic.capacity": {
        "family": "mechanism_diagnostic",
        "purpose": "Use only validated tangent capture to distinguish low, high, and unresolved capacity evidence.",
        "inputs": ["tangent_status", "capture_fraction"],
        "definition": "PASS tangent with capture fraction below 0.5 is low; at least 0.5 is high; otherwise unresolved",
        "operator": "<",
        "tolerance": 0.5,
        "cadence": "after tangent protocol",
        "patience": 1,
        "initialization": "tangent unresolved",
        "reset": "never after diagnostic freeze",
        "terminal": "nonterminal diagnostic",
        "precedence": [],
        "event": ["CAPACITY_DIAGNOSTIC"],
        "log": ["tangent_status", "capture_fraction", "classification"],
        "artifact": ["capacity_diagnostic.json"],
        "outcome": "may support DECODER_PARAMETERIZATION_CAPACITY_BARRIER only with complete evidence",
        "authorization": "capacity ladder only when the full authorization conjunction passes",
        "required": "required",
    },
    "safety.runtime_contract": {
        "family": "runtime_safety",
        "purpose": "Fail closed on any declared safety, hash, finiteness, fallback, output, mutation, or optimizer event.",
        "inputs": ["explicit_safety_event_booleans"],
        "definition": "any of the first ten stop-precedence events is a safety failure",
        "operator": "any == true",
        "tolerance": 0.0,
        "cadence": "launch, load authorization, every step, every evaluation, persistence, and shutdown",
        "patience": 1,
        "initialization": "clear",
        "reset": "never after a safety failure",
        "terminal": "all safety failures are immediate terminal",
        "precedence": ["first ten terminal-precedence events"],
        "event": ["RUNTIME_SAFETY_FAILURE"],
        "log": ["all_present_safety_events"],
        "artifact": ["runtime_safety_event.json"],
        "outcome": "IMPLEMENTATION_OR_CONTRACT_FAILURE or NO_SCIENTIFIC_RESULT depending on trajectory existence",
        "authorization": "none",
        "required": "required",
    },
    "persistence.artifact_integrity": {
        "family": "artifact_integrity",
        "purpose": "Require exclusive state creation, collision refusal, not-reached entries, canonical hashes, and replay.",
        "inputs": ["persistence_mode", "fresh_run_root", "semantic_state_record", "canonical_hash"],
        "definition": "append-only payloads; atomic manifest revision only inside fresh run; exact hash replay",
        "operator": "exact path and hash equality",
        "tolerance": 0.0,
        "cadence": "every state write and final replay",
        "patience": 1,
        "initialization": "empty fresh state root and revision zero",
        "reset": "never; append-only",
        "terminal": "collision, overwrite, or replay mismatch is a contract failure",
        "precedence": ["HISTORICAL_WRITE_ATTEMPT", "TARGET_OR_HASH_MISMATCH"],
        "event": ["ARTIFACT_PERSISTED", "ARTIFACT_COLLISION_REFUSED", "ARTIFACT_REPLAY_PASS"],
        "log": ["state", "path", "sha256", "manifest_revision"],
        "artifact": ["d3_state_machine_manifest.json", "semantic_state_persistence_tests.csv"],
        "outcome": "persistence defect is IMPLEMENTATION_OR_CONTRACT_FAILURE",
        "authorization": "no authorization until replay passes",
        "required": "required",
    },
}


def implementation_hashes() -> dict[str, str]:
    """Hash the exact executable source of every policy function."""

    return {
        policy_id: hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
        for policy_id, function in IMPLEMENTATION_FUNCTIONS.items()
    }


def build_policy_registry(fixture_map: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, Any]:
    """Build the canonical registry from the frozen specs and executed fixture names."""

    hashes = implementation_hashes()
    records: list[dict[str, Any]] = []
    for policy_id in POLICY_IDS:
        spec = POLICY_SPECS[policy_id]
        fixtures = fixture_map[policy_id]
        record = {
            "canonical_policy_id": policy_id,
            "policy_family": spec["family"],
            "semantic_version": POLICY_VERSION,
            "purpose": spec["purpose"],
            "required_inputs": spec["inputs"],
            "prohibited_inputs": COMMON_PROHIBITED,
            "metric_or_event_definition": spec["definition"],
            "threshold_or_exact_condition": spec["definition"],
            "comparison_operator": spec["operator"],
            "numerical_tolerance": spec["tolerance"],
            "evaluation_cadence": spec["cadence"],
            "rolling_window_or_patience": spec["patience"],
            "initialization_behavior": spec["initialization"],
            "reset_behavior": spec["reset"],
            "terminal_or_nonterminal_status": spec["terminal"],
            "stop_precedence": spec["precedence"],
            "event_code": spec["event"],
            "required_log_payload": spec["log"],
            "required_persisted_artifact": spec["artifact"],
            "outcome_implication": spec["outcome"],
            "downstream_authorization_implication": spec["authorization"],
            "implementation_function": f"src.d3_policy_engine.{IMPLEMENTATION_FUNCTIONS[policy_id].__name__}",
            "implementation_code_hash": hashes[policy_id],
            "synthetic_fixtures_exercising_it": list(fixtures["positive"]),
            "negative_fixtures": list(fixtures["negative"]),
            "required_or_optional_status": spec["required"],
        }
        records.append(record)
    registry = {
        "schema_version": "thayer-d3-policy-registry-v3",
        "semantic_version": POLICY_VERSION,
        "policy_count": len(records),
        "canonical_policy_ids": list(POLICY_IDS),
        "policies": records,
        "no_implicit_defaults": True,
    }
    validate_policy_registry(registry, verify_implementation=True)
    return registry


def _walk_finite(value: Any, policy_id: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PolicyContractError(policy_id, "registry contains a nonfinite number")
    if isinstance(value, dict):
        for child in value.values():
            _walk_finite(child, policy_id)
    elif isinstance(value, list):
        for child in value:
            _walk_finite(child, policy_id)


def validate_policy_registry(registry: Mapping[str, Any], verify_implementation: bool) -> frozenset[str]:
    """Validate exact record fields, IDs, versions, fixtures, and code hashes."""

    if set(registry) != {"schema_version", "semantic_version", "policy_count", "canonical_policy_ids", "policies", "no_implicit_defaults"}:
        raise PolicyContractError("registry.policy_set", "registry root fields differ from schema")
    if registry["schema_version"] != "thayer-d3-policy-registry-v3" or registry["semantic_version"] != POLICY_VERSION:
        raise PolicyContractError("registry.policy_set", "unknown policy registry version")
    if registry["no_implicit_defaults"] is not True:
        raise PolicyContractError("registry.policy_set", "implicit defaults are prohibited")
    policies = registry["policies"]
    if not isinstance(policies, list):
        raise PolicyContractError("registry.policy_set", "policies must be a list")
    identifiers = tuple(record.get("canonical_policy_id") for record in policies)
    missing = [value for value in POLICY_IDS if value not in identifiers or value not in registry["canonical_policy_ids"]]
    if missing:
        raise PolicyContractError(missing[0], "canonical policy is missing")
    if registry["policy_count"] != len(POLICY_IDS) or len(policies) != len(POLICY_IDS):
        raise PolicyContractError("registry.policy_set", "policy count mismatch")
    if identifiers != POLICY_IDS or tuple(registry["canonical_policy_ids"]) != POLICY_IDS:
        raise PolicyContractError(missing[0] if missing else "registry.policy_set", "canonical policy ID set/order mismatch")
    hashes = implementation_hashes() if verify_implementation is True else {}
    for record in policies:
        policy_id = record["canonical_policy_id"]
        if tuple(record) != RECORD_FIELDS:
            raise PolicyContractError(policy_id, "policy record fields differ from exact schema")
        if record["semantic_version"] != POLICY_VERSION:
            raise PolicyContractError(policy_id, "unknown policy version")
        if not isinstance(record["required_inputs"], list) or len(record["required_inputs"]) == 0:
            raise PolicyContractError(policy_id, "required inputs missing")
        if not isinstance(record["prohibited_inputs"], list) or "implicit_python_truthiness" not in record["prohibited_inputs"]:
            raise PolicyContractError(policy_id, "prohibited input contract incomplete")
        if not isinstance(record["synthetic_fixtures_exercising_it"], list) or len(record["synthetic_fixtures_exercising_it"]) == 0:
            raise PolicyContractError(policy_id, "positive fixtures missing")
        if not isinstance(record["negative_fixtures"], list) or len(record["negative_fixtures"]) == 0:
            raise PolicyContractError(policy_id, "negative fixtures missing")
        if verify_implementation is True and record["implementation_code_hash"] != hashes[policy_id]:
            raise PolicyContractError(policy_id, "implementation code hash mismatch")
        spec = POLICY_SPECS[policy_id]
        expected_values = {
            "policy_family": spec["family"],
            "semantic_version": POLICY_VERSION,
            "purpose": spec["purpose"],
            "required_inputs": spec["inputs"],
            "prohibited_inputs": COMMON_PROHIBITED,
            "metric_or_event_definition": spec["definition"],
            "threshold_or_exact_condition": spec["definition"],
            "comparison_operator": spec["operator"],
            "numerical_tolerance": spec["tolerance"],
            "evaluation_cadence": spec["cadence"],
            "rolling_window_or_patience": spec["patience"],
            "initialization_behavior": spec["initialization"],
            "reset_behavior": spec["reset"],
            "terminal_or_nonterminal_status": spec["terminal"],
            "stop_precedence": spec["precedence"],
            "event_code": spec["event"],
            "required_log_payload": spec["log"],
            "required_persisted_artifact": spec["artifact"],
            "outcome_implication": spec["outcome"],
            "downstream_authorization_implication": spec["authorization"],
            "implementation_function": f"src.d3_policy_engine.{IMPLEMENTATION_FUNCTIONS[policy_id].__name__}",
            "required_or_optional_status": spec["required"],
        }
        for field, expected in expected_values.items():
            if record[field] != expected:
                raise PolicyContractError(policy_id, f"frozen registry field changed: {field}")
        _walk_finite(record, policy_id)
    return frozenset(identifiers)


def policy_registry_schema() -> dict[str, Any]:
    """Return the strict schema persisted beside the registry."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "thayer-d3-policy-registry-v3.schema.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "semantic_version", "policy_count", "canonical_policy_ids", "policies", "no_implicit_defaults"],
        "properties": {
            "schema_version": {"const": "thayer-d3-policy-registry-v3"},
            "semantic_version": {"const": POLICY_VERSION},
            "policy_count": {"const": len(POLICY_IDS)},
            "canonical_policy_ids": {"type": "array", "prefixItems": [{"const": value} for value in POLICY_IDS], "items": False},
            "no_implicit_defaults": {"const": True},
            "policies": {
                "type": "array",
                "minItems": len(POLICY_IDS),
                "maxItems": len(POLICY_IDS),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(RECORD_FIELDS),
                    "properties": {field: {} for field in RECORD_FIELDS},
                },
            },
        },
    }

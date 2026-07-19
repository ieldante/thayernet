"""Typed constants and records for the executable Thayer-D3 policy contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


POLICY_VERSION = "3.0.0"
POLICY_IDS = (
    "control.expert_activity",
    "control.expert_death",
    "control.prompt_collapse",
    "diagnostic.tangent_protocol",
    "outcome.scientific_mapping",
    "state.semantic_persistence",
    "control.stop_event_precedence",
    "authorization.downstream",
    "control.success_gate",
    "control.budget_exhaustion",
    "diagnostic.assignment",
    "diagnostic.square_mapping",
    "diagnostic.optimization",
    "diagnostic.capacity",
    "safety.runtime_contract",
    "persistence.artifact_integrity",
)

OUTCOME_CATEGORIES = (
    "L0_FULL_DECODER_SUCCESS",
    "DECODER_OPTIMIZATION_BARRIER",
    "DECODER_PARAMETERIZATION_CAPACITY_BARRIER",
    "HARD_ASSIGNMENT_BARRIER",
    "SQUARE_MAPPING_OPTIMIZATION_BARRIER",
    "MIXED_CAUSE",
    "MECHANISM_UNRESOLVED",
    "IMPLEMENTATION_OR_CONTRACT_FAILURE",
    "NO_SCIENTIFIC_RESULT",
)

SEMANTIC_STATES = (
    "initial",
    "one_step",
    "lowest_objective",
    "closest_to_d1",
    "first_own_coverage",
    "first_alternate_coverage",
    "first_both_mode_coverage",
    "success",
    "terminal_failure",
    "budget_exhausted",
    "final",
)

STOP_PRECEDENCE = (
    ("ACCESS_GUARD_VIOLATION", 70),
    ("HISTORICAL_WRITE_ATTEMPT", 71),
    ("PROTECTED_PATH_ACCESS", 72),
    ("CACHE_BYTECODE_DELETE_EVENT", 73),
    ("TARGET_OR_HASH_MISMATCH", 74),
    ("NONFINITE", 75),
    ("MPS_FALLBACK", 76),
    ("PHYSICAL_NEGATIVE_OUTPUT", 77),
    ("CACHED_FEATURE_MUTATION", 78),
    ("OPTIMIZER_CONTRACT_VIOLATION", 79),
    ("EXPERT_DEAD", 80),
    ("PROMPT_COLLAPSE", 81),
    ("SUCCESS_GATE", 0),
    ("BUDGET_EXHAUSTED", 2),
)

NUMERICAL_ZERO = 1e-7
IMAGE_FLOOR = 1e-12
EXPERT_DEATH_PATIENCE = 3
PROMPT_COLLAPSE_PATIENCE = 3
SUCCESS_PATIENCE = 3
MAXIMUM_STEPS = 5000
TANGENT_RELATIVE_STEPS = (0.001, 0.0003, 0.0001)
TANGENT_RELATIVE_TOLERANCE = 0.0001
TANGENT_ABSOLUTE_FLOOR = 1e-12
TANGENT_PROBE_COUNT = 8
TANGENT_SEED = 20260713
TANGENT_MAXIMUM_EVALUATIONS = 64
TANGENT_TENSOR_ROLES = (
    "expert_1.final_head",
    "expert_2.final_head",
    "prompt_a.expert_1_penultimate",
    "prompt_a.expert_2_penultimate",
    "prompt_b.expert_1_penultimate",
    "prompt_b.expert_2_penultimate",
)


class PolicyContractError(RuntimeError):
    """A fail-closed policy-contract error carrying a canonical policy ID."""

    def __init__(self, policy_id: str, message: str):
        super().__init__(f"{policy_id}: {message}")
        self.policy_id = policy_id
        self.message = message


@dataclass(frozen=True)
class ExpertMetrics:
    expert_id: str
    evaluation_index: int
    learning_rate: float
    optimizer_member: bool
    parameter_finite: bool
    gradient_finite: bool
    raw_output_finite: bool
    physical_output_finite: bool
    gradient_norm: float
    parameter_update_norm: float
    physical_output_change_norm: float
    frozen_parameter_count: int = 0


@dataclass(frozen=True)
class ExpertActivityDecision:
    expert_id: str
    state: str
    inactivity_streak: int
    event_code: str | None
    terminal: bool
    branch_ids: tuple[str, ...]


@dataclass(frozen=True)
class PromptMetrics:
    evaluation_index: int
    prompt_pair_complete: bool
    expert_1_same_requested_distance: float
    expert_1_same_companion_distance: float
    expert_2_same_requested_distance: float
    expert_2_same_companion_distance: float
    canonical_source_swap_distance: float
    set_permutation_match: bool
    ordinary_scene_concentration: bool


@dataclass(frozen=True)
class PromptDecision:
    state: str
    collapse_streak: int
    event_code: str | None
    terminal: bool
    valid_source_swap: bool
    branch_ids: tuple[str, ...]


@dataclass(frozen=True)
class TangentEvidence:
    enabled: bool
    trajectory_frozen: bool
    checkpoint_frozen: bool
    primary_outcome_frozen: bool
    finite_baseline: bool
    jvp_available: bool
    vjp_available: bool
    maximum_relative_error: float | None
    sign_match: bool
    scale_match: bool
    precision_sufficient: bool
    prohibited_condition_number_claim: bool
    primary_outcome: str
    capture_fraction: float | None = None


@dataclass(frozen=True)
class TangentDecision:
    status: str
    valid: bool
    terminal: bool
    usable_for_capacity_authorization: bool
    branch_ids: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeEvidence:
    implementation_or_contract_failure: bool
    authoritative_trajectory_exists: bool
    full_scientific_success: bool
    optimization_barrier_supported: bool
    capacity_barrier_supported: bool
    hard_assignment_barrier_supported: bool
    square_mapping_barrier_supported: bool
    evidence_consistent: bool
    capacity_relies_on_tangent: bool = False
    tangent_protocol_passed: bool = False


@dataclass(frozen=True)
class AuthorizationContext:
    outcome: str
    authoritative_d3_scientific_failure: bool
    d0_authoritative_pass: bool
    d1_authoritative_pass: bool
    prompt_gate_pass: bool
    forward_gate_pass: bool
    fresh_process_replay_pass: bool
    contract_unchanged: bool
    contract_integrity: bool
    code_runtime_loss_mapping_assignment_defect: bool
    tangent_evidence_used: bool
    tangent_protocol_passed: bool


@dataclass(frozen=True)
class PolicyDecision:
    policy_id: str
    status: str
    event_code: str | None = None
    terminal: bool = False
    branch_ids: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FixtureResult:
    fixture_id: int
    name: str
    status: str
    policy_ids: tuple[str, ...]
    branch_ids: tuple[str, ...]
    assertions: Mapping[str, Any]


@dataclass(frozen=True)
class SemanticCandidate:
    state: str
    evaluation_index: int
    step_index: int
    payload: bytes
    scalar_metrics: Mapping[str, float | int | bool | str | None]
    optimizer_state_sha256: str | None
    assignment: Mapping[str, Any]
    event: Mapping[str, Any]
    terminal_status: str
    objective: float | None = None
    distance_to_d1: float | None = None
    semantic_members: Sequence[str] = ()

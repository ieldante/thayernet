"""Explicit mode-aware encoder/cached-feature execution contract for D3."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class D3ExecutionMode(str, Enum):
    """Frozen D3 execution modes; no implicit/default mode exists."""

    CACHED_FEATURE_DECODER = "CACHED_FEATURE_DECODER"
    END_TO_END_ENCODER = "END_TO_END_ENCODER"


@dataclass(frozen=True)
class EncoderExecutionEvidence:
    """Typed immutable evidence supplied by the frozen execution contract."""

    execution_mode: object
    encoder_constructed: bool
    encoder_forward_count: int
    encoder_parameter_count: Optional[int]
    encoder_trainable_parameter_count: int
    encoder_optimizer_parameter_count: int
    cached_features_loaded: bool
    cached_feature_container_count: int
    cached_feature_member_count: int
    cached_feature_hashes_match: bool
    cached_feature_schema_contracts_pass: bool
    cached_features_finite: bool
    cached_feature_prompt_identities_pass: bool
    cached_feature_expert_semantics_pass: bool
    decoder_constructed: bool
    decoder_parameter_contract_passes: bool
    optimizer_constructed: bool
    optimizer_parameter_contract_passes: bool
    encoder_forward_in_scope: bool
    optimizer_in_scope: bool
    expected_encoder_parameter_count: Optional[int]
    expected_encoder_trainable_parameter_count: Optional[int]
    expected_encoder_optimizer_parameter_count: Optional[int]


@dataclass(frozen=True)
class EncoderExecutionContractResult:
    """Immutable result with complete machine- and human-readable evidence."""

    execution_mode: str
    passed: bool
    mode_specific_required_conditions: Tuple[str, ...]
    condition_results: Tuple[Tuple[str, bool], ...]
    failed_condition_ids: Tuple[str, ...]
    verified_safety_properties: Tuple[str, ...]
    reason: str
    status_code: str

    def as_record(self) -> dict:
        return {
            "execution_mode": self.execution_mode,
            "passed": self.passed,
            "mode_specific_required_conditions": list(self.mode_specific_required_conditions),
            "condition_results": {name: passed for name, passed in self.condition_results},
            "failed_condition_ids": list(self.failed_condition_ids),
            "verified_safety_properties": list(self.verified_safety_properties),
            "reason": self.reason,
            "status_code": self.status_code,
        }


def _result(
    *,
    mode: str,
    conditions: Tuple[Tuple[str, bool], ...],
    safety: Tuple[str, ...],
    success_code: str,
    failure_code: str,
) -> EncoderExecutionContractResult:
    failed = tuple(name for name, passed in conditions if not passed)
    passed = not failed
    reason = (
        f"{mode} execution contract passed"
        if passed
        else f"{mode} execution contract failed: {', '.join(failed)}"
    )
    return EncoderExecutionContractResult(
        execution_mode=mode,
        passed=passed,
        mode_specific_required_conditions=tuple(name for name, _ in conditions),
        condition_results=conditions,
        failed_condition_ids=failed,
        verified_safety_properties=safety,
        reason=reason,
        status_code=success_code if passed else failure_code,
    )


def evaluate_encoder_execution_contract(
    evidence: EncoderExecutionEvidence,
) -> EncoderExecutionContractResult:
    """Evaluate only the explicitly supplied mode; unknown values fail closed."""

    if evidence.execution_mode is D3ExecutionMode.CACHED_FEATURE_DECODER:
        conditions = (
            ("encoder_not_constructed", evidence.encoder_constructed is False),
            ("encoder_forward_count_zero", evidence.encoder_forward_count == 0),
            (
                "encoder_parameter_count_zero_or_absent",
                evidence.encoder_parameter_count in (None, 0),
            ),
            (
                "encoder_trainable_parameter_count_zero",
                evidence.encoder_trainable_parameter_count == 0,
            ),
            (
                "encoder_optimizer_parameter_count_zero",
                evidence.encoder_optimizer_parameter_count == 0,
            ),
            ("cached_features_loaded", evidence.cached_features_loaded is True),
            (
                "cached_feature_container_count_positive",
                evidence.cached_feature_container_count > 0,
            ),
            (
                "cached_feature_member_count_positive",
                evidence.cached_feature_member_count > 0,
            ),
            (
                "cached_feature_hashes_match",
                evidence.cached_feature_hashes_match is True,
            ),
            (
                "cached_feature_schema_contracts_pass",
                evidence.cached_feature_schema_contracts_pass is True,
            ),
            ("cached_features_finite", evidence.cached_features_finite is True),
            (
                "cached_feature_prompt_identities_pass",
                evidence.cached_feature_prompt_identities_pass is True,
            ),
            (
                "cached_feature_expert_semantics_pass",
                evidence.cached_feature_expert_semantics_pass is True,
            ),
            ("decoder_constructed", evidence.decoder_constructed is True),
            (
                "decoder_parameter_contract_passes",
                evidence.decoder_parameter_contract_passes is True,
            ),
            (
                "optimizer_constructed_when_in_scope",
                not evidence.optimizer_in_scope or evidence.optimizer_constructed is True,
            ),
            (
                "optimizer_parameter_contract_passes",
                not evidence.optimizer_in_scope
                or evidence.optimizer_parameter_contract_passes is True,
            ),
        )
        passed_by_name = dict(conditions)
        safety = tuple(
            property_id
            for property_id, verified in (
                ("ENCODER_ABSENCE_VERIFIED", passed_by_name["encoder_not_constructed"]),
                (
                    "ENCODER_FORWARD_COUNT_ZERO",
                    passed_by_name["encoder_forward_count_zero"],
                ),
                (
                    "ENCODER_OPTIMIZER_PARAMETER_COUNT_ZERO",
                    passed_by_name["encoder_optimizer_parameter_count_zero"],
                ),
                (
                    "FROZEN_CACHED_FEATURES_VERIFIED",
                    all(
                        passed_by_name[name]
                        for name in (
                            "cached_features_loaded",
                            "cached_feature_container_count_positive",
                            "cached_feature_member_count_positive",
                            "cached_feature_hashes_match",
                            "cached_feature_schema_contracts_pass",
                            "cached_features_finite",
                            "cached_feature_prompt_identities_pass",
                            "cached_feature_expert_semantics_pass",
                        )
                    ),
                ),
            )
            if verified
        )
        return _result(
            mode=D3ExecutionMode.CACHED_FEATURE_DECODER.value,
            conditions=conditions,
            safety=safety,
            success_code="CACHED_FEATURE_EXECUTION_CONTRACT_PASS",
            failure_code="CACHED_FEATURE_EXECUTION_CONTRACT_FAIL",
        )

    if evidence.execution_mode is D3ExecutionMode.END_TO_END_ENCODER:
        conditions = (
            ("encoder_constructed", evidence.encoder_constructed is True),
            (
                "encoder_forward_count_positive",
                not evidence.encoder_forward_in_scope or evidence.encoder_forward_count > 0,
            ),
            (
                "encoder_parameter_contract_explicit",
                evidence.expected_encoder_parameter_count is not None,
            ),
            (
                "encoder_parameter_count_matches",
                evidence.expected_encoder_parameter_count is not None
                and evidence.encoder_parameter_count
                == evidence.expected_encoder_parameter_count,
            ),
            (
                "encoder_trainable_parameter_count_matches",
                evidence.expected_encoder_trainable_parameter_count is not None
                and evidence.encoder_trainable_parameter_count
                == evidence.expected_encoder_trainable_parameter_count,
            ),
            (
                "cached_feature_substitution_prohibited",
                evidence.cached_features_loaded is False
                and evidence.cached_feature_container_count == 0
                and evidence.cached_feature_member_count == 0,
            ),
            ("decoder_constructed", evidence.decoder_constructed is True),
            (
                "decoder_parameter_contract_passes",
                evidence.decoder_parameter_contract_passes is True,
            ),
            (
                "optimizer_constructed_when_in_scope",
                not evidence.optimizer_in_scope or evidence.optimizer_constructed is True,
            ),
            (
                "optimizer_parameter_contract_passes",
                not evidence.optimizer_in_scope
                or evidence.optimizer_parameter_contract_passes is True,
            ),
            (
                "encoder_optimizer_parameter_count_matches",
                not evidence.optimizer_in_scope
                or (
                    evidence.expected_encoder_optimizer_parameter_count is not None
                    and evidence.encoder_optimizer_parameter_count
                    == evidence.expected_encoder_optimizer_parameter_count
                ),
            ),
        )
        return _result(
            mode=D3ExecutionMode.END_TO_END_ENCODER.value,
            conditions=conditions,
            safety=(),
            success_code="END_TO_END_EXECUTION_CONTRACT_PASS",
            failure_code="END_TO_END_EXECUTION_CONTRACT_FAIL",
        )

    supplied = getattr(evidence.execution_mode, "value", evidence.execution_mode)
    return EncoderExecutionContractResult(
        execution_mode=str(supplied),
        passed=False,
        mode_specific_required_conditions=("recognized_explicit_execution_mode",),
        condition_results=(("recognized_explicit_execution_mode", False),),
        failed_condition_ids=("recognized_explicit_execution_mode",),
        verified_safety_properties=(),
        reason=f"unknown explicit D3 execution mode: {supplied!r}",
        status_code="UNKNOWN_EXECUTION_MODE",
    )


__all__ = [
    "D3ExecutionMode",
    "EncoderExecutionEvidence",
    "EncoderExecutionContractResult",
    "evaluate_encoder_execution_contract",
]

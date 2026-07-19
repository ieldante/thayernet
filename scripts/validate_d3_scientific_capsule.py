#!/usr/bin/env python3
"""Fail-closed validator for the Thayer-D3 scientific contract capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "thayer-d3-scientific-capsule-v1"
SEMANTICS_VERSION = "thayer-d3-scientific-semantics-v1"
COMPLETENESS_MARKER = "ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED"
PLACEHOLDERS = {"TODO", "UNKNOWN", "TBD", "NONE", "NULL", "IMPLICIT_DEFAULT"}


class CapsuleValidationError(RuntimeError):
    """One or more capsule invariants failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    rows = [(prefix, value)]
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_walk(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_walk(item, f"{prefix}[{index}]"))
    return rows


def _schema_validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(instance, dict),
        "array": isinstance(instance, list),
        "string": isinstance(instance, str),
        "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "boolean": isinstance(instance, bool),
    }.get(expected_type, True)
    if not type_ok:
        return [f"SCHEMA_TYPE:{path}:{expected_type}"]
    if "const" in schema and instance != schema["const"]:
        errors.append(f"SCHEMA_CONST:{path}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"SCHEMA_ENUM:{path}")
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in instance:
                errors.append(f"SCHEMA_REQUIRED:{path}.{required}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"SCHEMA_ADDITIONAL:{path}.{key}")
        for key, subschema in properties.items():
            if key in instance:
                errors.extend(_schema_validate(instance[key], subschema, f"{path}.{key}"))
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"SCHEMA_MIN_ITEMS:{path}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"SCHEMA_MAX_ITEMS:{path}")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in instance}) != len(instance):
            errors.append(f"SCHEMA_UNIQUE:{path}")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors.extend(_schema_validate(item, schema["items"], f"{path}[{index}]"))
    return errors


def validate_capsule(
    capsule: dict[str, Any],
    schema: dict[str, Any],
    *,
    repo: Path | None,
    verify_files: bool,
) -> list[str]:
    errors = _schema_validate(capsule, schema)
    identity = capsule.get("capsule_identity", {})
    semantics = capsule.get("scientific_semantics", {})
    observation = capsule.get("observation_configuration", {})
    forward = capsule.get("forward_plausibility", {})
    coverage = capsule.get("truth_coverage", {})
    tolerances = capsule.get("numerical_tolerances", {})
    implementation = capsule.get("implementation_hashes", {})
    artifacts = capsule.get("scientific_artifact_references", {})
    runtime = capsule.get("runtime_contract", {})
    completeness = capsule.get("completeness", {})

    if identity.get("schema_version") != SCHEMA_VERSION:
        errors.append("UNKNOWN_REQUIRED_SCHEMA_VERSION")
    if semantics.get("version") != SEMANTICS_VERSION:
        errors.append("UNKNOWN_REQUIRED_SEMANTICS_VERSION")
    if semantics.get("bands") != ["g", "r", "z"]:
        errors.append("BAND_ORDER_MISMATCH")
    if semantics.get("channel_order") != [
        "requested_g",
        "requested_r",
        "requested_z",
        "companion_g",
        "companion_r",
        "companion_z",
    ]:
        errors.append("CHANNEL_ORDER_MISMATCH")
    if semantics.get("implicit_defaults_permitted") is not False:
        errors.append("IMPLICIT_DEFAULT_PRESENT")

    sky = observation.get("scientific_sky_vector", {})
    sky_values = sky.get("values")
    if not isinstance(sky_values, list) or len(sky_values) != 3:
        errors.append("SKY_VECTOR_SHAPE_MISMATCH")
    elif any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in sky_values):
        errors.append("SKY_VECTOR_INVALID")
    if sky.get("band_order") != ["g", "r", "z"]:
        errors.append("SKY_VECTOR_BAND_ORDER_MISMATCH")
    if sky.get("unit") != "detected electrons per pixel":
        errors.append("SKY_VECTOR_UNIT_MISMATCH")
    if sky.get("semantic_name") != "per_band_additive_sky_electron_expectation_for_poisson_variance":
        errors.append("SKY_VECTOR_SEMANTICS_MISMATCH")

    plausibility = forward.get("thresholds", {})
    required_plausibility = {
        "global_chi_square_mean",
        "per_band_chi_square_mean",
        "absolute_relative_flux_residual",
    }
    if set(plausibility) != required_plausibility:
        errors.append("PLAUSIBILITY_THRESHOLD_MISSING_OR_EXTRA")
    if forward.get("comparison_operators") != {
        "global_chi_square_mean": "<=",
        "per_band_chi_square_mean": "<=",
        "absolute_relative_flux_residual": "<=",
    }:
        errors.append("PLAUSIBILITY_OPERATOR_MISMATCH")
    per_band = plausibility.get("per_band_chi_square_mean")
    if not isinstance(per_band, dict) or set(per_band) != {"g", "r", "z"}:
        errors.append("PER_BAND_PLAUSIBILITY_THRESHOLD_MISMATCH")
    for _, value in _walk(plausibility):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append("NONFINITE_THRESHOLD")

    truth_thresholds = coverage.get("thresholds", {})
    if truth_thresholds.get("image_symmetric_relative_l2") != 0.25:
        errors.append("IMAGE_THRESHOLD_MISMATCH")
    if truth_thresholds.get("relative_flux_by_band") != {"g": 0.2, "r": 0.2, "z": 0.2}:
        errors.append("FLUX_THRESHOLD_MISMATCH")
    if truth_thresholds.get("color_magnitude") != {"g-r": 0.2, "r-z": 0.2}:
        errors.append("COLOR_THRESHOLD_MISMATCH")
    centroid = truth_thresholds.get("centroid", {})
    if centroid.get("value") != 0.5 or centroid.get("unit") != "mean PSF FWHM":
        errors.append("CENTROID_THRESHOLD_OR_UNIT_MISMATCH")
    if truth_thresholds.get("ordinary_concentration_primary_diameter") != 1.0:
        errors.append("ORDINARY_CONCENTRATION_THRESHOLD_MISMATCH")
    if coverage.get("comparison_operator") != "<=":
        errors.append("TRUTH_COVERAGE_OPERATOR_MISMATCH")

    if tolerances.get("numerical_zero_normalized") != 1e-7:
        errors.append("NUMERICAL_ZERO_TOLERANCE_MISMATCH")
    if tolerances.get("physical_negative_detected_electrons") != 0.0:
        errors.append("PHYSICAL_NEGATIVE_TOLERANCE_MISMATCH")
    if tolerances.get("finite_value_nonfinite_count") != 0:
        errors.append("FINITE_VALUE_TOLERANCE_MISMATCH")
    if tolerances.get("physical_roundtrip_atol_detected_electrons") != 0.00390625:
        errors.append("ROUNDTRIP_TOLERANCE_MISMATCH")

    if semantics.get("output_mapping", {}).get("identifier") != "square":
        errors.append("OUTPUT_MAPPING_MISMATCH")
    mapping_hash = semantics.get("output_mapping", {}).get("code_sha256")
    if mapping_hash != implementation.get("mapping", {}).get("sha256"):
        errors.append("MAPPING_HASH_INTERNAL_MISMATCH")
    evaluator_hash = forward.get("evaluator_sha256")
    if evaluator_hash != implementation.get("pure_forward_evaluator", {}).get("sha256"):
        errors.append("EVALUATOR_HASH_INTERNAL_MISMATCH")

    if completeness.get("marker") != COMPLETENESS_MARKER:
        errors.append("COMPLETENESS_MARKER_MISSING")
    required_count = completeness.get("required_field_count")
    if required_count != completeness.get("resolved_field_count"):
        errors.append("DEPENDENCY_COUNT_MISMATCH")
    if completeness.get("unresolved_field_count") != 0:
        errors.append("UNRESOLVED_DEPENDENCY")
    if completeness.get("conflict_count") != 0:
        errors.append("PROVENANCE_CONFLICT")
    provenance = capsule.get("provenance", {}).get("field_records", [])
    if not isinstance(required_count, int) or len(provenance) != required_count:
        errors.append("PROVENANCE_FIELD_COUNT_MISMATCH")

    for path, value in _walk(capsule):
        if value is None:
            errors.append(f"NULL_REQUIRED_VALUE:{path}")
        if isinstance(value, str) and value.strip().upper() in PLACEHOLDERS:
            errors.append(f"PLACEHOLDER:{path}")
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"NONFINITE:{path}")

    for name, reference in artifacts.items():
        relative = str(reference.get("relative_path", ""))
        lowered = relative.lower()
        if "atlas" in lowered or "development" in lowered or "lockbox" in lowered:
            errors.append(f"PROTECTED_RUNTIME_ARTIFACT_PATH:{name}")

    for value in runtime.get("prohibited_runtime_dependencies", []):
        if value not in {"Atlas", "development", "lockbox", "historical scientific configuration lookup"}:
            errors.append("UNKNOWN_PROHIBITED_RUNTIME_DEPENDENCY")
    if runtime.get("historical_configuration_lookup_permitted") is not False:
        errors.append("HISTORICAL_CONFIGURATION_LOOKUP_ENABLED")

    if verify_files:
        if repo is None:
            errors.append("REPO_REQUIRED_FOR_FILE_VERIFICATION")
        else:
            repo = repo.resolve()
            for name, record in implementation.items():
                relative = record.get("relative_path")
                if relative:
                    path = repo / relative
                    if not path.is_file() or sha256(path) != record.get("sha256"):
                        errors.append(f"IMPLEMENTATION_HASH_MISMATCH:{name}")
            for name, record in artifacts.items():
                path = repo / record.get("relative_path", "")
                if (
                    not path.is_file()
                    or path.stat().st_size != record.get("bytes")
                    or sha256(path) != record.get("sha256")
                ):
                    errors.append(f"ARTIFACT_HASH_MISMATCH:{name}")
            runtime_manifest = runtime.get("runtime_readiness_manifest", {})
            runtime_path = repo / runtime_manifest.get("relative_path", "")
            if not runtime_path.is_file() or sha256(runtime_path) != runtime_manifest.get("sha256"):
                errors.append("RUNTIME_MANIFEST_HASH_MISMATCH")
    return sorted(set(errors))


def validate_files(
    *,
    repo: Path,
    capsule_path: Path,
    schema_path: Path,
    manifest_path: Path | None = None,
    hash_chain_path: Path | None = None,
) -> dict[str, Any]:
    capsule = _json(capsule_path)
    schema = _json(schema_path)
    errors = validate_capsule(capsule, schema, repo=repo, verify_files=True)
    if manifest_path is not None:
        manifest = _json(manifest_path)
        if manifest.get("capsule_sha256") != sha256(capsule_path):
            errors.append("MANIFEST_CAPSULE_HASH_MISMATCH")
        if manifest.get("schema_sha256") != sha256(schema_path):
            errors.append("MANIFEST_SCHEMA_HASH_MISMATCH")
        if hash_chain_path is not None and manifest.get("hash_chain_sha256") != sha256(hash_chain_path):
            errors.append("MANIFEST_HASH_CHAIN_MISMATCH")
    if hash_chain_path is not None:
        chain = _json(hash_chain_path)
        for name, record in chain.get("entries", {}).items():
            path = repo / record.get("relative_path", "")
            if (
                not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256(path) != record.get("sha256")
            ):
                errors.append(f"HASH_CHAIN_MISMATCH:{name}")
    errors = sorted(set(errors))
    return {
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "capsule_sha256": sha256(capsule_path),
        "schema_sha256": sha256(schema_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hash-chain", type=Path)
    args = parser.parse_args()
    result = validate_files(
        repo=args.repo,
        capsule_path=args.capsule,
        schema_path=args.schema,
        manifest_path=args.manifest,
        hash_chain_path=args.hash_chain,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

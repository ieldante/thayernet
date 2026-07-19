"""Canonical registry helpers shared by every executable D3 component."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REGISTRY_SCHEMA_VERSION = "thayer-d3-requirement-registry-v2"
REQUIRED_RECORD_FIELDS = (
    "canonical_requirement_id",
    "human_readable_name",
    "category",
    "required",
    "data_type",
    "expected_shape",
    "expected_dtype",
    "units",
    "semantic_version",
    "capsule_location",
    "source_provenance_requirement",
    "consumers",
    "validation_function",
    "representation_kind",
    "scientific_deserialization_required",
    "protected_data_restrictions",
    "failure_message",
    "expected_value",
)


@dataclass(frozen=True)
class RequirementFailure(RuntimeError):
    requirement_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.requirement_id}: {self.message}"


def _parse_json_or_string(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def records_from_v1_inventory(path: Path) -> list[dict[str, object]]:
    """Translate the authoritative 97-row dependency inventory into v2 records."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 97:
        raise RequirementFailure("registry.v1_scientific_requirement_count", f"expected 97 rows, found {len(rows)}")
    records: list[dict[str, object]] = []
    for row in rows:
        identifier = row["canonical_field_name"]
        representation = "embedded small value"
        if row["data_type"] == "artifact_reference":
            representation = "file reference"
        elif identifier.startswith("implementation."):
            representation = "code hash"
        records.append(make_record(
            identifier,
            human_name=row["semantic_purpose"],
            category=row["category"],
            data_type=row["data_type"],
            expected_shape=row["shape"],
            expected_dtype=row["data_type"],
            units=row["units"],
            value=_parse_json_or_string(row["value"]),
            provenance={
                "source_artifact": row["source_artifact"],
                "source_key": row["source_key"],
                "source_sha256": row["source_sha256"],
                "extraction_method": row["extraction_method"],
                "independent_confirmation": row["independent_confirmation"],
                "classification": row["classification"],
            },
            consumers=[row["consumer"]],
            representation=representation,
            scientific_deserialization=identifier.startswith("artifact."),
        ))
    return records


def make_record(
    identifier: str,
    *,
    human_name: str,
    category: str,
    data_type: str,
    expected_shape: object,
    expected_dtype: object,
    units: str,
    value: object,
    provenance: object,
    consumers: list[str],
    representation: str,
    scientific_deserialization: bool = False,
    validation_function: str = "validate_expected_value_exact",
    restrictions: str = "No development, lockbox, Atlas scene, ordinary, eight-scene, or full-microset access.",
) -> dict[str, object]:
    return {
        "canonical_requirement_id": identifier,
        "human_readable_name": human_name,
        "category": category,
        "required": True,
        "data_type": data_type,
        "expected_shape": expected_shape,
        "expected_dtype": expected_dtype,
        "units": units,
        "semantic_version": "2.0.0",
        "capsule_location": f"requirements.{identifier}",
        "source_provenance_requirement": provenance,
        "consumers": consumers,
        "validation_function": validation_function,
        "representation_kind": representation,
        "scientific_deserialization_required": scientific_deserialization,
        "protected_data_restrictions": restrictions,
        "failure_message": f"D3 requirement failed: {identifier}",
        "expected_value": value,
    }


def build_registry(records: Iterable[Mapping[str, object]], *, created_utc: str) -> dict[str, object]:
    requirements = [dict(record) for record in records]
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_id": "thayer-d3-executable-contract-v2",
        "created_utc": created_utc,
        "sole_authoritative_declaration": True,
        "requirements": sorted(requirements, key=lambda item: str(item["canonical_requirement_id"])),
    }
    validate_registry(registry)
    return registry


def validate_registry(registry: Mapping[str, object]) -> None:
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RequirementFailure("registry.schema_version", "wrong registry schema version")
    if registry.get("sole_authoritative_declaration") is not True:
        raise RequirementFailure("registry.sole_authoritative_declaration", "registry is not marked sole authoritative")
    requirements = registry.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise RequirementFailure("registry.requirements", "requirements must be a nonempty list")
    identifiers: list[str] = []
    for record in requirements:
        if not isinstance(record, dict):
            raise RequirementFailure("registry.record_type", "every requirement must be an object")
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise RequirementFailure(str(record.get("canonical_requirement_id", "registry.record_fields")), f"missing record fields {missing}")
        identifier = record["canonical_requirement_id"]
        if not isinstance(identifier, str) or not identifier:
            raise RequirementFailure("registry.canonical_requirement_id", "identifier must be nonempty")
        if record["required"] is not True:
            raise RequirementFailure(identifier, "all v2 registry records must be required")
        if not isinstance(record["consumers"], list) or not record["consumers"]:
            raise RequirementFailure(identifier, "consumer list must be nonempty")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise RequirementFailure("registry.unique_requirement_ids", "duplicate canonical requirement ID")


def records_by_id(registry: Mapping[str, object]) -> dict[str, dict[str, object]]:
    validate_registry(registry)
    return {
        str(record["canonical_requirement_id"]): dict(record)
        for record in registry["requirements"]  # type: ignore[index]
    }


def required_ids(registry: Mapping[str, object]) -> frozenset[str]:
    return frozenset(records_by_id(registry))


def required_ids_for_component(registry: Mapping[str, object], component: str) -> frozenset[str]:
    """All components intentionally consume the one full required set."""

    if component not in {
        "builder", "validator", "metadata_preflight", "model_preflight",
        "synthetic_consumer", "scientific_launcher", "future_preregistration",
    }:
        raise ValueError(component)
    return required_ids(registry)


_PLACEHOLDER_TOKENS = {"", "tbd", "todo", "placeholder", "unknown", "unresolved", "missing"}


def _assert_no_placeholder(identifier: str, value: object) -> None:
    if value is None:
        raise RequirementFailure(identifier, "null is prohibited")
    if isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_TOKENS:
        raise RequirementFailure(identifier, f"placeholder is prohibited: {value!r}")
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_placeholder(identifier, item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_placeholder(identifier, item)


def validate_capsule_requirements(
    capsule: Mapping[str, object],
    registry: Mapping[str, object],
    *,
    accessed: set[str] | None = None,
) -> frozenset[str]:
    expected_records = records_by_id(registry)
    actual = capsule.get("requirements")
    if not isinstance(actual, dict):
        raise RequirementFailure("registry.capsule_requirements_mapping", "capsule requirements mapping is absent")
    declared = set(expected_records)
    actual_ids = set(actual)
    extras = sorted(actual_ids - declared)
    if extras:
        raise RequirementFailure("registry.no_undeclared_requirement", f"undeclared requirement entries: {extras}")
    missing = sorted(declared - actual_ids)
    if missing:
        raise RequirementFailure(missing[0], "required capsule entry is absent")
    for identifier in sorted(declared):
        value = actual[identifier]
        _assert_no_placeholder(identifier, value)
        expected = expected_records[identifier]["expected_value"]
        if value != expected:
            raise RequirementFailure(identifier, "capsule value does not exactly match registry")
        if accessed is not None:
            accessed.add(identifier)
    return frozenset(declared)


def registry_value(registry: Mapping[str, object], identifier: str, accessed: set[str] | None = None) -> object:
    record = records_by_id(registry).get(identifier)
    if record is None:
        raise RequirementFailure("registry.no_undeclared_requirement", f"consumer requested undeclared dependency {identifier}")
    if accessed is not None:
        accessed.add(identifier)
    return record["expected_value"]

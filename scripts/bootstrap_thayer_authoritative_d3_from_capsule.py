#!/usr/bin/env python3
"""Capsule-only preflight; validates readiness and exits without running D3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from d3_capsule_evaluator_selftest import run_capsule_evaluator_tests
from validate_d3_scientific_capsule import validate_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hash-chain", type=Path, required=True)
    args = parser.parse_args()

    result = validate_files(
        repo=args.repo,
        capsule_path=args.capsule,
        schema_path=args.schema,
        manifest_path=args.manifest,
        hash_chain_path=args.hash_chain,
    )
    if result["status"] != "PASS":
        raise SystemExit("capsule validation failed: " + "; ".join(result["errors"]))
    capsule = json.loads(args.capsule.read_text(encoding="utf-8"))
    rows = run_capsule_evaluator_tests(capsule, args.repo.resolve())
    if len(rows) != 12 or any(row["status"] != "PASS" for row in rows):
        raise SystemExit("capsule evaluator self-tests failed")
    if capsule["completeness"]["marker"] != "ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED":
        raise SystemExit("capsule completeness marker mismatch")
    print("ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED")
    print("READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

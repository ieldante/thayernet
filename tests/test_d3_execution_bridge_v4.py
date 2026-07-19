"""Authority-chain and schema tests for execution bridge v4."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.d3_execution_bridge_v4 import (
    CAPSULE_V1_SHA256,
    CONTINUATION_VALUE,
    IntegrationRequirementFailure,
    bridge_schema,
    build_bridge,
    validate_authority_chain,
)


REPO = Path(__file__).resolve().parents[1]
V3 = REPO / "outputs/runs/thayer_d3_policy_contract_20260713_173955/bundle_v3/d3_executable_bundle_v3.json"
V3_SHA = "30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c"


class ExecutionBridgeV4Tests(unittest.TestCase):
    def test_authority_chain_resolves_v2_only_from_v3(self) -> None:
        chain = validate_authority_chain(REPO, V3, V3_SHA)
        self.assertEqual(chain["bundle_v3"]["base_bundle_v2"]["sha256"], "884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045")
        self.assertEqual(chain["capsule_v1_path"].name, "d3_scientific_capsule_v1.json")
        self.assertEqual(CAPSULE_V1_SHA256, "8a76ccdfa659a7291f0f9b73e0cb4d4c8adfb317b9902fc8ad5763e6d17b7d21")

    def test_wrong_bundle_v3_hash_rejected_with_id(self) -> None:
        with self.assertRaises(IntegrationRequirementFailure) as raised:
            validate_authority_chain(REPO, V3, "0" * 64)
        self.assertEqual(raised.exception.requirement_id, "D3I-AUTH-BUNDLE-V3-SHA")

    def test_bridge_is_reference_only_and_freezes_precedence(self) -> None:
        bridge = build_bridge(
            repo=REPO,
            run=REPO / "outputs/runs/test-never-written",
            bundle_v3_path=V3,
            bundle_v3_sha256=V3_SHA,
            repository_head="74b8ff7efbbf7e9891cc8fd8095a9931e3b63174",
            phase="candidate",
            synthetic_preflight={"status": "TEST"},
        )
        self.assertEqual(bridge["precedence"]["automatic_scientific_continuation"], CONTINUATION_VALUE)
        self.assertFalse(bridge["precedence"]["bundle_v2_historical_scope_authoritative"])
        serialized = json.dumps(bridge)
        self.assertNotIn("scientific_sky_vector", serialized)
        self.assertNotIn("normalization_scale_grz", serialized)

    def test_schema_requires_all_bridge_sections(self) -> None:
        schema = bridge_schema()
        self.assertEqual(schema["properties"]["schema_version"]["const"], "thayer-d3-execution-bridge-v4")
        self.assertIn("flow_invariants", schema["required"])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from scripts.audit_thayer_select_hierarchical_protocol import reconstruct_rows, truth_table


class OriginalContractAuditTests(unittest.TestCase):
    def test_truth_table_is_complete_and_unique(self):
        table = truth_table()
        self.assertEqual(len(table), 140)
        self.assertFalse(table.duplicated().any())
        self.assertEqual(set(table.query_state), {"UNIQUE_VALID", "NULL", "AMBIGUOUS"})

    def test_query_specific_applicability_and_labels(self):
        common = {
            "partition": "training",
            "reconstruction_provenance": "test",
            "evaluation_valid": True,
            "predicted_to_blend_flux_ratio": 0.01,
            "hallucination": False,
            "forced_source_selection": False,
            "source_confusion": False,
            "normalized_rmse": 0.1,
            "max_relative_flux_error": 0.1,
            "max_color_error_mag": 0.1,
            "centroid_error_pixels": 0.1,
            "catastrophic_failure": False,
        }
        frame = pd.DataFrame([
            {**common, "scene_id": "valid", "query_class": "VALID_SOURCE", "moderate_success": 1, "moderate_actionable_success": 1},
            {**common, "scene_id": "null", "query_class": "NULL_SOURCE", "max_color_error_mag": np.nan, "centroid_error_pixels": np.nan, "moderate_success": 1, "moderate_actionable_success": 0},
            {**common, "scene_id": "ambiguous", "query_class": "AMBIGUOUS_SOURCE", "max_color_error_mag": np.nan, "centroid_error_pixels": np.nan, "moderate_success": 0, "moderate_actionable_success": 0},
        ])
        audited = reconstruct_rows(frame).set_index("scene_id")
        self.assertEqual(int(audited.loc["valid", "original_composite_label"]), 1)
        self.assertEqual(int(audited.loc["null", "original_contract_success"]), 1)
        self.assertEqual(int(audited.loc["null", "original_composite_label"]), 0)
        self.assertEqual(audited.loc["null", "image_pass"], "NA")
        self.assertEqual(int(audited.loc["ambiguous", "composite_logically_meaningful_for_query"]), 0)
        self.assertTrue((audited.label_formula_match == 1).all())
        self.assertTrue((audited.contract_formula_match == 1).all())


if __name__ == "__main__":
    unittest.main()

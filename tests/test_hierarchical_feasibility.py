import math

import numpy as np

from src.hierarchical_feasibility import (
    SEMANTICS,
    ambiguous_forced_output,
    catastrophic_valid_failure,
    null_hallucination_outcomes,
)
from src.hierarchical_safety import QueryState, associate_hierarchical_query


def test_matching_radius_boundary_is_inclusive():
    result = associate_hierarchical_query(np.array([[10.0, 10.0]]), np.array([14.0, 10.0]), image_shape=(60, 60), semantics=SEMANTICS)
    assert result.state is QueryState.UNIQUE_VALID


def test_just_outside_matching_radius_is_null():
    result = associate_hierarchical_query(np.array([[10.0, 10.0]]), np.array([14.0 + 1e-9, 10.0]), image_shape=(60, 60), semantics=SEMANTICS)
    assert result.state is QueryState.NULL


def test_ambiguity_margin_boundary_is_inclusive_and_tie_has_no_truth():
    sources = np.array([[10.0, 10.0], [13.0, 10.0]])
    result = associate_hierarchical_query(sources, np.array([11.0, 10.0]), image_shape=(60, 60), semantics=SEMANTICS)
    assert result.state is QueryState.AMBIGUOUS
    assert result.matched_index is None


def test_stable_tie_handling_does_not_assign_source():
    sources = np.array([[9.0, 10.0], [11.0, 10.0]])
    result = associate_hierarchical_query(sources, np.array([10.0, 10.0]), image_shape=(60, 60), semantics=SEMANTICS)
    assert result.state is QueryState.AMBIGUOUS
    assert result.matched_index is None


def test_inclusive_image_edge_and_outside_rejection():
    source = np.array([[0.0, 0.0]])
    assert associate_hierarchical_query(source, np.array([0.0, 0.0]), image_shape=(60, 60), semantics=SEMANTICS).state is QueryState.UNIQUE_VALID
    try:
        associate_hierarchical_query(source, np.array([-1e-9, 0.0]), image_shape=(60, 60), semantics=SEMANTICS)
    except ValueError:
        pass
    else:
        raise AssertionError("outside prompt must fail")


def test_perturbed_valid_tolerance_is_inside_matching_radius():
    assert SEMANTICS.maximum_perturbation_pixels < SEMANTICS.matching_radius_pixels


def test_catastrophic_boundary_and_nonfinite_fail_closed():
    assert catastrophic_valid_failure(image_risk=1.5, flux_risk_max=0.0, centroid_risk_pixels=0.0, confusion=False)
    assert catastrophic_valid_failure(image_risk=0.0, flux_risk_max=0.0, centroid_risk_pixels=0.0, confusion=True)
    assert catastrophic_valid_failure(image_risk=math.nan, flux_risk_max=0.0, centroid_risk_pixels=0.0, confusion=False)
    assert not catastrophic_valid_failure(image_risk=1.5 - 1e-9, flux_risk_max=1.0 - 1e-9, centroid_risk_pixels=4.0 - 1e-9, confusion=False)


def test_null_hallucination_preserves_continuous_outcomes():
    blend = np.ones((3, 4, 4), dtype=float)
    zero = null_hallucination_outcomes(np.zeros_like(blend), blend)
    assert zero["null_hallucination"] is False
    assert zero["null_output_energy_ratio"] == 0.0
    exposed = null_hallucination_outcomes(0.2 * blend, blend)
    assert exposed["null_hallucination"] is True
    assert exposed["null_absolute_flux_ratio"] == 0.2


def test_ambiguous_forced_output_is_descriptive_without_truth_assignment():
    sources = np.stack((np.zeros((3, 4, 4)), np.ones((3, 4, 4))))
    result = ambiguous_forced_output(sources[1], sources, np.ones((3, 4, 4)))
    assert result["ambiguous_forced_output"] is True
    assert result["exposed_source_index"] == 1

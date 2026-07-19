import numpy as np

from src.conditional_calibration import (
    apply_tertiles,
    attainable_prevalence_relative_threshold,
    conformal_quantile,
    conformal_rank,
    crossfit_bounds,
    deployable_mondrian_group,
    fixed_tertile_edges,
    group_safe_folds,
    order_statistic_resolution,
    verify_fold_isolation,
)


def test_prevalence_relative_gate_is_attainable():
    threshold = attainable_prevalence_relative_threshold(0.8165, 0.75)
    assert 0.8165 < threshold < 1.0
    assert np.isclose(threshold, 0.954125)


def test_conformal_order_statistic_and_resolution():
    assert conformal_rank(9, 0.9) == 9
    assert conformal_rank(100, 0.9) == 91
    assert np.isclose(order_statistic_resolution(100), 1 / 101)
    values = np.arange(100, dtype=float)
    assert conformal_quantile(values, 0.9) == 90.0


def test_frozen_tertiles_replay_without_result_tuning():
    train = np.arange(1, 301, dtype=float)
    edges = fixed_tertile_edges(train)
    labels = apply_tertiles(np.array([1.0, 150.0, 300.0]), edges, ("low", "medium", "high"))
    assert labels.tolist() == ["low", "medium", "high"]


def test_source_groups_do_not_cross_folds():
    source_a = np.array(["a", "b", "d", "f", "g"])
    source_b = np.array(["b", "c", "e", "g", "h"])
    folds = group_safe_folds(source_a, source_b, folds=3)
    assert verify_fold_isolation(source_a, source_b, folds)
    assert folds[0] == folds[1]
    assert folds[3] == folds[4]


def test_normalized_conformal_uses_scale_without_nonfinite_bounds():
    n = 50
    residual = np.linspace(-0.2, 0.6, n)
    central = np.linspace(0.1, 1.0, n)
    scale = np.linspace(0.1, 0.5, n)
    features = np.column_stack([central, scale])
    fold = np.arange(n) % 5
    groups = deployable_mondrian_group(central, scale, fixed_tertile_edges(central), float(np.median(scale)))
    bound, support = crossfit_bounds(
        residual, central, scale, features, fold, "C2_normalized", groups, minimum_support=3, neighbors=5
    )
    assert np.isfinite(bound).all()
    assert (support == 40).all()


def test_all_conditional_methods_return_one_bound_per_row():
    rng = np.random.default_rng(7)
    n = 100
    residual = rng.normal(0.0, 0.3, n)
    central = rng.normal(1.0, 0.1, n)
    scale = rng.uniform(0.1, 1.0, n)
    features = rng.normal(size=(n, 4))
    fold = np.arange(n) % 5
    groups = deployable_mondrian_group(central, scale, fixed_tertile_edges(central), float(np.median(scale)))
    for method in ("C0_global", "C1_mondrian", "C2_normalized", "C3_local", "C4_mondrian_normalized"):
        bound, support = crossfit_bounds(
            residual, central, scale, features, fold, method, groups, minimum_support=3, neighbors=10
        )
        assert bound.shape == (n,)
        assert support.shape == (n,)
        assert np.isfinite(bound).all()
        assert (support > 0).all()

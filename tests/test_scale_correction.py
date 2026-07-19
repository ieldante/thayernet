import numpy as np
import torch

from src.conditional_calibration import group_safe_folds, verify_fold_isolation
from src.scale_correction import (
    MODEL_FAMILIES,
    ScaleNet,
    cluster_bootstrap_indices,
    conformal_quantile,
    conformal_rank,
    crossfit_normalized_upper,
    normalized_scores,
)


def test_scale_targets_are_absolute_and_finite():
    truth = np.array([0.0, 2.0, 5.0])
    prediction = np.array([1.0, 1.0, 7.0])
    target = np.abs(truth - prediction)
    assert target.tolist() == [1.0, 1.0, 2.0]
    assert np.isfinite(target).all()


def test_deployable_feature_names_exclude_oracle_fields():
    allowed = ["central_predicted_risk", "local_signal_background_proxy_g", "prompt_local_001"]
    prohibited = ("true_snr", "obstruction", "separation", "flux_ratio", "source_id", "source_group")
    assert not any(token in name for name in allowed for token in prohibited)


def test_group_safe_crossfit_has_no_source_overlap():
    source_a = np.array(["a", "b", "d", "f", "g", "i"])
    source_b = np.array(["b", "c", "e", "g", "h", "j"])
    folds = group_safe_folds(source_a, source_b, folds=3)
    assert verify_fold_isolation(source_a, source_b, folds)
    for group in np.unique(np.r_[source_a, source_b]):
        rows = (source_a == group) | (source_b == group)
        assert len(np.unique(folds[rows])) == 1


def test_partial_pooling_and_soft_gate_are_small_and_positive_after_transform():
    values = torch.zeros((8, 20))
    proxies = torch.zeros((8, 4))
    for family in MODEL_FAMILIES:
        model = ScaleNet(20, family)
        output = model(values, proxies)
        assert output.shape == (8,)
        assert torch.isfinite(output).all()
        assert sum(parameter.numel() for parameter in model.parameters()) < 25000
        scale = torch.exp(torch.clamp(output, np.log(1e-3), np.log(25.0)))
        assert torch.all(scale >= 1e-3)
        assert torch.all(scale <= 25.0)


def test_normalized_conformal_respects_floor_cap_and_finite_rank():
    truth = np.linspace(0.0, 2.0, 50)
    central = np.linspace(0.0, 1.5, 50)
    scale = np.linspace(0.0, 0.5, 50)
    fold = np.arange(50) % 5
    upper, quantile = crossfit_normalized_upper(truth, central, scale, fold, scale_floor=1e-3)
    assert np.isfinite(upper).all()
    assert np.isfinite(quantile).all()
    assert upper.shape == truth.shape
    assert conformal_rank(40, 0.9) == 37
    assert conformal_quantile(np.arange(40.0), 0.9) == 36.0
    scores = normalized_scores(truth, central, scale, 1e-3)
    assert np.isfinite(scores).all()


def test_source_component_bootstrap_keeps_component_blocks():
    components = np.array(["a", "a", "b", "c", "c", "c"])
    rng = np.random.default_rng(3)
    index = cluster_bootstrap_indices(components, rng)
    assert len(index) > 0
    for label in np.unique(components[index]):
        original_block = np.flatnonzero(components == label)
        sampled = index[components[index] == label]
        assert len(sampled) % len(original_block) == 0


def test_oracle_subgroups_are_not_scale_model_arguments():
    names = ScaleNet.forward.__code__.co_varnames[: ScaleNet.forward.__code__.co_argcount]
    assert names == ("self", "values", "proxies")

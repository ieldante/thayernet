import math

import numpy as np
import torch

from src.direct_catalog_safety_auditor import (
    PostAuditSafetyNetwork,
    PreAuditQueryNetwork,
    SCALAR_FEATURE_NAMES,
    binary_auprc,
    binary_auroc,
    connected_components,
    deployable_scalar_features,
    fit_binary_temperature,
    fit_multiclass_temperature,
    normalized_post_image,
    normalized_pre_image,
    policy_metrics,
    post_audit_supervision,
    select_fail_closed_threshold,
    softmax,
    trainable_parameter_count,
)


def arrays():
    blend = np.zeros((3, 60, 60), dtype=np.float32)
    truth = np.zeros_like(blend)
    alternate = np.zeros_like(blend)
    truth[:, 28:33, 28:33] = np.asarray([10.0, 8.0, 6.0])[:, None, None]
    alternate[:, 10:14, 10:14] = 3.0
    blend = truth + alternate
    prompt = np.zeros((1, 60, 60), dtype=np.float32)
    prompt[0, 30, 30] = 1.0
    return blend, prompt, truth, alternate


def test_architecture_parameter_ceilings_and_shapes():
    pre = PreAuditQueryNetwork()
    post = PostAuditSafetyNetwork()
    assert trainable_parameter_count(pre) < 100_000
    assert trainable_parameter_count(post) < 350_000
    assert pre(torch.zeros(2, 4, 60, 60)).shape == (2, 3)
    assert post(torch.zeros(2, 10, 60, 60), torch.zeros(2, len(SCALAR_FEATURE_NAMES))).shape == (2,)


def test_deployable_feature_schema_and_image_channels():
    blend, prompt, truth, _ = arrays()
    features = deployable_scalar_features(blend, prompt, truth)
    assert features.shape == (len(SCALAR_FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert normalized_pre_image(blend, prompt, [10, 10, 10]).shape == (4, 60, 60)
    assert normalized_post_image(blend, prompt, truth, [10, 10, 10]).shape == (10, 60, 60)


def test_truth_only_changes_label_not_deployable_features():
    blend, prompt, truth, alternate = arrays()
    before = deployable_scalar_features(blend, prompt, truth)
    changed_truth = truth * 3
    after = deployable_scalar_features(blend, prompt, truth)
    assert np.array_equal(before, after)
    assert post_audit_supervision(truth, truth, alternate, blend).unsafe_to_catalog is False
    assert post_audit_supervision(truth, changed_truth, alternate, blend).unsafe_to_catalog is True


def test_physical_output_and_false_subtraction_are_unsafe():
    blend, _, truth, alternate = arrays()
    negative = truth.copy(); negative[:, 0, 0] = -1e-5
    assert post_audit_supervision(negative, truth, alternate, blend).physical_output_contract_failure
    zero = np.zeros_like(truth)
    label = post_audit_supervision(zero, truth, alternate, blend)
    assert label.false_subtraction_failure
    assert label.unsafe_to_catalog


def test_binary_metrics_and_temperature():
    truth = np.asarray([0, 0, 1, 1])
    score = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert binary_auroc(score, truth) == 1.0
    assert binary_auprc(score, truth) == 1.0
    temperature = fit_binary_temperature(np.asarray([-2.0, -1.0, 1.0, 2.0]), truth)
    assert math.isfinite(temperature) and temperature > 0
    logits = np.asarray([[3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=float)
    multi_temperature = fit_multiclass_temperature(logits, np.asarray([0, 1, 2]))
    assert multi_temperature > 0
    assert np.allclose(softmax(logits).sum(axis=1), 1.0)


def test_threshold_attainability_and_fail_closed_behavior():
    query = np.asarray([0, 0, 1, 2])
    pre = np.asarray([0, 0, 0, 0])
    post = np.asarray([0.1, 0.9, 0.1, 0.1])
    unsafe = np.asarray([0, 1, 0, 0])
    catastrophic = unsafe.copy()
    threshold, metrics, feasible, _ = select_fail_closed_threshold(query, pre, post, unsafe, catastrophic)
    assert not feasible
    assert threshold < np.min(post)
    assert metrics["accepted_coverage"] == 0.0
    direct = policy_metrics(query, np.asarray([0, 1, 1, 2]), post, unsafe, catastrophic, 0.5)
    assert direct["accepted_coverage"] == 0.5


def test_connected_source_groups_do_not_split_components():
    component = connected_components([("a", "b"), ("b", "c"), ("d", "e")])
    assert component[0] == component[1]
    assert component[0] != component[2]


def test_mps_execution_if_available():
    if not torch.backends.mps.is_available():
        return
    device = torch.device("mps")
    pre = PreAuditQueryNetwork().to(device)
    post = PostAuditSafetyNetwork().to(device)
    assert pre(torch.zeros(1, 4, 60, 60, device=device)).device.type == "mps"
    assert post(
        torch.zeros(1, 10, 60, 60, device=device),
        torch.zeros(1, len(SCALAR_FEATURE_NAMES), device=device),
    ).device.type == "mps"

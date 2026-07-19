import numpy as np
import torch

from src.shape_constrained_quantile import (
    ConvexQuantileScale,
    constraint_diagnostics,
    fit_shape_model,
    pinball_loss,
)


def fixture_model(interaction: bool = True) -> ConvexQuantileScale:
    rng = np.random.default_rng(7)
    proxy = rng.uniform(0, 1, size=(128, 4)).astype(np.float32)
    knots = np.quantile(proxy, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0).T
    return ConvexQuantileScale(knots, proxy, scale_floor=0.001, scale_cap=5.0, interaction=interaction)


def test_pinball_loss_is_zero_at_exact_target():
    value = torch.tensor([1.0, 2.0])
    assert pinball_loss(value, value).item() == 0.0


def test_main_effects_are_centered_on_training_rows():
    rng = np.random.default_rng(8)
    proxy = rng.uniform(0, 1, size=(256, 4)).astype(np.float32)
    knots = np.quantile(proxy, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0).T
    model = ConvexQuantileScale(knots, proxy, scale_floor=0.001, scale_cap=5.0, interaction=True)
    with torch.no_grad():
        model.start.add_(torch.tensor([0.7, -0.2, 1.1, -0.5]))
        model.delta_raw.add_(torch.linspace(-0.8, 0.9, 20).reshape(4, 5))
    centered = model.main_effects(torch.from_numpy(proxy)).detach().numpy()
    assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-6)


def test_constraints_and_positive_product_interaction():
    model = fixture_model(True)
    fit = type("Fit", (), {"model": model})()
    diagnostics = constraint_diagnostics(fit)
    assert diagnostics["convexity_violations"] == 0
    assert diagnostics["upper_half_monotonicity_violations"] == 0
    assert diagnostics["interaction_nonnegative"]
    points = torch.tensor([[0.4, 0.9, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0]])
    effect = model.interaction_effect(points).detach().numpy()
    assert effect[0] == 0.0
    assert effect[1] > 0.0


def test_outputs_respect_floor_and_cap():
    model = fixture_model(True)
    proxy = torch.tensor([[0.0] * 4, [1.0] * 4])
    scale = model(proxy).detach().numpy()
    assert np.all(scale >= 0.001)
    assert np.all(scale <= 5.0)


def test_tiny_fit_is_finite_and_under_parameter_ceiling():
    rng = np.random.default_rng(9)
    train = rng.uniform(0, 1, size=(256, 4)).astype(np.float32)
    validation = rng.uniform(0, 1, size=(64, 4)).astype(np.float32)
    train_y = np.abs(train[:, 0] - 0.5).astype(np.float32) + 0.1
    validation_y = np.abs(validation[:, 0] - 0.5).astype(np.float32) + 0.1
    knots = np.quantile(train, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0).T
    fit = fit_shape_model(
        "Q2", 11, train, train_y, validation, validation_y, knots,
        scale_floor=0.001, scale_cap=5.0, max_epochs=3, patience=2,
    )
    assert np.isfinite(fit.best_validation_loss)
    assert sum(parameter.numel() for parameter in fit.model.parameters()) <= 64

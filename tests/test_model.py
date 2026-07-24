import torch
import pytest

from psanet.model import PSANet, PSANetConfig
from psanet.losses import quantile_loss


@pytest.mark.parametrize("n_features", [3, 6, 20])
def test_shapes_generic_across_feature_counts(n_features):
    """The architecture must produce correct shapes for any n_features,
    with no code changes — this is what makes it a reusable config, not a
    one-off script hardcoded to one dataset."""
    cfg = PSANetConfig(
        n_features=n_features, input_window=1440, forecast_horizon=60,
        patch_len=60, patch_stride=30, d_model=64, n_heads=4,
        n_layers=2, n_quantiles=3, steps_per_day=1440,
    )
    model = PSANet(cfg)
    B = 4
    x = torch.randn(B, cfg.input_window, cfg.n_features)
    n_patches = model.patch_embed.n_patches
    tod_idx = torch.randint(0, cfg.steps_per_day, (B, n_patches))
    dow_idx = torch.randint(0, cfg.days_per_week, (B, n_patches))

    out = model(x, tod_idx, dow_idx)
    assert out.shape == (B, cfg.forecast_horizon, cfg.n_features, cfg.n_quantiles)


def test_gradients_flow():
    """A backward pass must reach every parameter — catches silently
    disconnected branches (e.g. a toggled-off module still being referenced)."""
    cfg = PSANetConfig(n_features=6, input_window=1440, forecast_horizon=60,
                        patch_len=60, patch_stride=30, d_model=64, n_heads=4,
                        n_layers=2, n_quantiles=3, steps_per_day=1440)
    model = PSANet(cfg)
    B = 4
    x = torch.randn(B, cfg.input_window, cfg.n_features)
    n_patches = model.patch_embed.n_patches
    tod_idx = torch.randint(0, cfg.steps_per_day, (B, n_patches))
    dow_idx = torch.randint(0, cfg.days_per_week, (B, n_patches))
    target = torch.randn(B, cfg.forecast_horizon, cfg.n_features)

    out = model(x, tod_idx, dow_idx)
    loss = quantile_loss(out, target, quantiles=[0.1, 0.5, 0.9])
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


@pytest.mark.parametrize("use_spectral,use_spike,use_seasonal", [
    (False, False, False),  # the documented fallback path — must still work
    (True, False, False),
    (False, True, False),
    (True, True, True),
])
def test_optional_branches_toggle_cleanly(use_spectral, use_spike, use_seasonal):
    """Every optional branch (spectral, spike bank, seasonal embed) must be
    independently toggleable without breaking shapes — this is the documented
    fallback path if the full model is unstable under deadline pressure."""
    cfg = PSANetConfig(
        n_features=6, input_window=1440, forecast_horizon=60, patch_len=60,
        patch_stride=30, d_model=64, n_heads=4, n_layers=2, n_quantiles=3,
        steps_per_day=1440, use_spectral_branch=use_spectral,
        use_spike_bank=use_spike, use_seasonal_embed=use_seasonal,
    )
    model = PSANet(cfg)
    B = 2
    x = torch.randn(B, cfg.input_window, cfg.n_features)
    n_patches = model.patch_embed.n_patches
    tod_idx = torch.randint(0, cfg.steps_per_day, (B, n_patches))
    dow_idx = torch.randint(0, cfg.days_per_week, (B, n_patches))

    out = model(x, tod_idx, dow_idx)
    assert out.shape == (B, cfg.forecast_horizon, cfg.n_features, cfg.n_quantiles)


def test_quantile_loss_zero_at_perfect_prediction():
    """Sanity check on the loss function itself, independent of the model."""
    target = torch.randn(2, 5, 3)
    quantiles = [0.1, 0.5, 0.9]
    preds = target.unsqueeze(-1).repeat(1, 1, 1, len(quantiles))
    loss = quantile_loss(preds, target, quantiles)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

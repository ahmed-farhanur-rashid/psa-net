import numpy as np
import pandas as pd
import pytest

from psanet.model import PSANetConfig
from psanet.dataset import TelemetryDataset, make_splits


def _make_df(n_rows=2000, n_features=3, freq="1min"):
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-01-01", periods=n_rows, freq=freq)
    data = {"timestamp": ts}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(size=n_rows)
    return pd.DataFrame(data)


def test_normalization_uses_train_stats_not_val_stats():
    """Validation split must reuse the training split's mean/std — reusing
    val's own statistics would leak information and understate error."""
    df = _make_df(n_rows=3000)
    cfg = PSANetConfig(n_features=3, input_window=200, forecast_horizon=20,
                        patch_len=20, patch_stride=10, steps_per_day=1440)
    feature_cols = ["feat_0", "feat_1", "feat_2"]

    train_ds, val_ds = make_splits(df, feature_cols, cfg, val_fraction=0.15)
    assert np.array_equal(train_ds.mean, val_ds.mean)
    assert np.array_equal(train_ds.std, val_ds.std)


def test_empty_split_raises_loudly_not_silently():
    """A too-small split must raise, not silently produce a 0-length dataset
    that a naive training loop would report as a meaningless 0.0 loss."""
    df = _make_df(n_rows=100)  # too small for input_window=1440 default
    cfg = PSANetConfig(n_features=3, input_window=1440, forecast_horizon=60,
                        patch_len=60, patch_stride=30, steps_per_day=1440)
    feature_cols = ["feat_0", "feat_1", "feat_2"]

    with pytest.raises(ValueError):
        make_splits(df, feature_cols, cfg, val_fraction=0.15)


def test_time_of_day_index_correct_at_1min_resolution():
    """Regression test for a real bug: step-of-day index must be derived
    from steps_per_day, not a hardcoded 5-minute assumption."""
    df = _make_df(n_rows=2000, freq="1min")
    cfg = PSANetConfig(n_features=3, input_window=200, forecast_horizon=20,
                        patch_len=20, patch_stride=10, steps_per_day=1440)
    ds = TelemetryDataset(df, ["feat_0", "feat_1", "feat_2"], cfg)

    # at 1-min resolution, minute 90 of the day should map to tod index 90
    # (row 90 is 2026-01-01 01:30:00 -> hour=1, minute=30 -> 60+30=90)
    assert ds.tod[90] == 90


def test_time_of_day_index_correct_at_5min_resolution():
    df = _make_df(n_rows=2000, freq="5min")
    cfg = PSANetConfig(n_features=3, input_window=200, forecast_horizon=20,
                        patch_len=20, patch_stride=10, steps_per_day=288)
    ds = TelemetryDataset(df, ["feat_0", "feat_1", "feat_2"], cfg)

    # at 5-min resolution, row 18 is 2026-01-01 01:30:00 -> step 18 of the day (90 min / 5)
    assert ds.tod[18] == 18


def test_getitem_shapes():
    df = _make_df(n_rows=2000)
    cfg = PSANetConfig(n_features=3, input_window=200, forecast_horizon=20,
                        patch_len=20, patch_stride=10, steps_per_day=1440)
    ds = TelemetryDataset(df, ["feat_0", "feat_1", "feat_2"], cfg)

    hist, tod_idx, dow_idx, target = ds[0]
    assert hist.shape == (200, 3)
    assert target.shape == (20, 3)
    assert tod_idx.shape[0] == ds.n_patches
    assert dow_idx.shape[0] == ds.n_patches

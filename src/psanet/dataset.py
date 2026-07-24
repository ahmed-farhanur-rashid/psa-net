"""Dataset and windowing logic for PSA-Net training."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from psanet.model import PSANetConfig


class TelemetryDataset(Dataset):
    """
    Windows a multivariate telemetry CSV into (history, time-of-day index,
    day-of-week index, target) tuples for training/eval.

    Normalization (z-score) is fit on whichever split constructs it first;
    pass mean/std explicitly when constructing a val/test split so it reuses
    the training split's statistics rather than leaking val/test information
    into normalization.
    """

    def __init__(self, df: pd.DataFrame, feature_cols, cfg: PSANetConfig, mean=None, std=None):
        self.cfg = cfg
        self.feature_cols = feature_cols
        values = df[feature_cols].values.astype(np.float32)

        if mean is None:
            mean = values.mean(axis=0)
            std = values.std(axis=0) + 1e-6
        self.mean, self.std = mean, std
        self.values = (values - mean) / std

        ts_col = "timestamp" if "timestamp" in df.columns else ("ds" if "ds" in df.columns else df.columns[0])
        dt = pd.to_datetime(df[ts_col])
        # step-of-day, generic over resolution: minutes-since-midnight / minutes-per-step.
        # minutes-per-step is derived from cfg.steps_per_day (1440 steps/day => 1 min/step;
        # 288 steps/day => 5 min/step), so this is correct for any resolution as long as
        # steps_per_day matches the actual data resolution.
        minutes_per_step = 1440 // cfg.steps_per_day
        self.tod = ((dt.dt.hour * 60 + dt.dt.minute) // minutes_per_step).values
        self.dow = dt.dt.dayofweek.values

        self.n_patches = (cfg.input_window - cfg.patch_len) // cfg.patch_stride + 1
        self.valid_starts = len(self.values) - cfg.input_window - cfg.forecast_horizon

    def __len__(self):
        return max(0, self.valid_starts)

    def __getitem__(self, idx):
        cfg = self.cfg
        hist = self.values[idx: idx + cfg.input_window]
        target = self.values[idx + cfg.input_window: idx + cfg.input_window + cfg.forecast_horizon]

        # tod/dow index at the START of each patch within the history window
        patch_starts = np.arange(0, self.n_patches) * cfg.patch_stride
        tod_idx = self.tod[idx + patch_starts] % cfg.steps_per_day
        dow_idx = self.dow[idx + patch_starts] % cfg.days_per_week

        return (
            torch.tensor(hist, dtype=torch.float32),
            torch.tensor(tod_idx, dtype=torch.long),
            torch.tensor(dow_idx, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
        )


def make_splits(df: pd.DataFrame, feature_cols, cfg: PSANetConfig, val_fraction: float = 0.15):
    """Chronological train/val split (no shuffling — this is time series) + fitted datasets."""
    n = len(df)
    split = int(n * (1 - val_fraction))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    train_ds = TelemetryDataset(train_df, feature_cols, cfg)
    val_ds = TelemetryDataset(val_df, feature_cols, cfg, mean=train_ds.mean, std=train_ds.std)

    if len(train_ds) == 0:
        raise ValueError(
            f"Training split has 0 usable windows (need >= input_window+horizon = "
            f"{cfg.input_window + cfg.forecast_horizon} rows, got {len(train_df)}). "
            f"Use more data or reduce input_window/horizon."
        )
    if len(val_ds) == 0:
        raise ValueError(
            f"Validation split has 0 usable windows (need >= input_window+horizon = "
            f"{cfg.input_window + cfg.forecast_horizon} rows, got {len(val_df)}). "
            f"Use more data, reduce input_window/horizon, or lower val_fraction."
        )
    return train_ds, val_ds

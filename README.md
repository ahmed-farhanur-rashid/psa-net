# PSA-Net — Patch-Spectral Attention Network

A configurable, attention-only architecture for multivariate time-series
forecasting. Combines patch tokenization, dual time/spectral (FFT) branches,
explicit seasonal embeddings, and a learned spike-pattern cross-attention
bank. Originally built for Kubernetes HPA telemetry forecasting; the
architecture is dataset-agnostic (config-driven, tested shape-generic across
3/6/20 feature counts).

See `src/psanet/model.py` docstrings for full design rationale on each
component.

## Install

```bash
pip install -e ".[dev]"
```

## Train

```bash
python -m psanet.train \
    --config configs/base.yaml \
    --csv data/02_processed/your_data.csv \
    --features requests_per_second cpu_utilization_pct gpu_utilization_pct \
    --out models/psanet_checkpoint.pt
```

Any field in `configs/base.yaml` can be overridden from the CLI, e.g.
`--epochs 50 --d_model 256 --forecast_horizon 30`.

**Data resolution:** `configs/base.yaml` defaults to 1-minute resolution
(`steps_per_day: 1440`). If your data is 5-minute, set `steps_per_day: 288`
and rescale `input_window`/`forecast_horizon`/`patch_len`/`patch_stride` by
5x (defaults are sized in real time-span, not step count). This must match
your actual data — it is not auto-detected, and a mismatch produces silently
wrong seasonal embeddings, not an error.

Expects a CSV with a `timestamp` column plus your feature columns.

## Test

```bash
pytest tests/ -v
```

14 tests covering: shape-genericity across feature counts, gradient flow,
the fallback-path configuration (all optional branches disabled), dataset
normalization correctness (train stats not leaked from val), the loud-failure
guard on undersized splits, and a regression test for time-of-day indexing
at both 1-min and 5-min resolution (this was a real bug once — a hardcoded
5-minute assumption silently corrupting seasonal embeddings at 1-min
resolution — the test exists specifically so it can't come back unnoticed).

## Fallback path

If the full model is unstable or slow to converge, disable the added
branches in the config and it degrades to a plain patch-attention forecaster
(PatchTST-equivalent) with no other code changes:

```yaml
use_spectral_branch: false
use_spike_bank: false
```

(`use_seasonal_embed` can stay `true` independently — it's the cheapest,
most reliable piece.)

## Repo structure

```
psanet/
├── configs/
│   └── base.yaml          # hyperparameters; override any field from the CLI
├── src/psanet/
│   ├── model.py            # architecture + PSANetConfig
│   ├── dataset.py          # windowing, normalization, train/val split
│   ├── losses.py           # quantile (pinball) loss
│   └── train.py            # training loop, config-driven
├── tests/
│   ├── test_model.py
│   └── test_dataset.py
├── notebooks/               # EDA / prototyping only — nothing production runs from here
├── models/                  # checkpoints land here (gitignored)
└── pyproject.toml
```

`data/` is intentionally not included yet — add it (with `.gitignore`
entries already in place) once there's a real, tracked dataset to put there.
Same for `.github/workflows/` — add CI once there's something concrete to
gate (e.g. "tests must pass before merge").

## Known limitation

The forecast head flattens all (feature × patch) tokens into one linear
layer, so parameter count scales roughly with `n_features²` and with
`forecast_horizon`. At the 1-minute-resolution defaults (6 features,
horizon=60): ~19.7M params. Fine at the intended scale; if pushing to
20+ features, replace the flatten-head with a per-feature output projection
sharing the backbone.

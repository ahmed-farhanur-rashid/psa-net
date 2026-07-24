# Model Card for PSA-Net (Patch-Spectral Attention Network)

## Model Summary

- **Model Name**: PSA-Net (Patch-Spectral Attention Network)
- **Model Type**: Configurable, Multi-Branch Deep Transformer for Multivariate Time-Series Forecasting
- **Framework**: PyTorch
- **License**: MIT
- **Primary Task**: Multi-step ahead, multi-quantile multivariate time-series forecasting

---

## Architectural Overview

PSA-Net is a dataset-agnostic architecture designed for time-series forecasting on metrics exhibiting high seasonality and sudden trend spikes. It integrates four core components:

1. **Patch Tokenization**:
   - Divides raw input sequences into overlapping time patches (e.g., `patch_len=15`, `patch_stride=15`).
   - Reduces token length and captures local temporal patterns.

2. **Dual Time / Spectral (FFT) Branch**:
   - Encodes each patch both in the raw time domain and in the frequency domain via Real Fast Fourier Transform (RFFT).
   - Fuses time and frequency features to give periodic seasonality dedicated representation paths.

3. **Explicit Seasonal Embeddings**:
   - High-resolution Time-of-Day and Day-of-Week embedding tables added per patch timestamp position.

4. **Spike Pattern Cross-Attention Bank**:
   - Learned memory bank of reference "spike shape" embeddings.
   - Patch tokens cross-attend to this bank to identify early signatures of sudden workload surges.

5. **Multi-Quantile Output Head**:
   - Joint linear projection layer predicting uncertainty quantiles ($q_{0.1}, q_{0.5}, q_{0.9}$) for all features across future timesteps in a single forward pass.

```
Input History [Batch, input_window, N_features]
       │
       ├──► Patch Tokenization (patch_len, patch_stride)
       ├──► FFT Spectral Branch (n_freq_bins)
       ├──► Time-of-Day / Day-of-Week Embeddings
       └──► Spike Pattern Cross-Attention Bank (n_spike_patterns)
       │
 Transformer Blocks (n_layers, d_model, n_heads)
       │
 Quantile Output Head
       │
       ▼
Forecast Tensor [Batch, forecast_horizon, N_features, n_quantiles]
```

---

## Technical Specifications & Config Parameters

| Hyperparameter | Default | Description |
| :--- | :--- | :--- |
| `n_features` | Configurable ($N$) | Number of input time-series signals |
| `input_window` | `120` | Length of historical input sequence (timesteps) |
| `forecast_horizon` | `15` or `60` | Number of future timesteps to predict |
| `patch_len` | `15` | Timesteps per patch |
| `patch_stride` | `15` | Stride between consecutive patches |
| `d_model` | `128` | Model hidden embedding dimension |
| `n_heads` | `4` | Number of Multi-Head Attention heads |
| `n_layers` | `3` | Number of Transformer encoder blocks |
| `use_spectral_branch` | `True` | Enables FFT frequency-domain fusion branch |
| `use_seasonal_embed` | `True` | Enables Time-of-Day and Day-of-Week embeddings |
| `use_spike_bank` | `True` | Enables Spike-Pattern Cross-Attention memory bank |
| `n_quantiles` | `3` | Number of output quantiles ($q_{0.1}, q_{0.5}, q_{0.9}$) |

---

## Direct Python Usage Examples

### 1. Model Initialization

```python
import torch
from psanet.model import PSANet, PSANetConfig

# 1. Define Model Configuration
cfg = PSANetConfig(
    n_features=6,           # Works for any N_features
    input_window=120,       # 2-hour lookback @ 1-min resolution
    forecast_horizon=15,    # 15-minute future forecast
    patch_len=15,
    patch_stride=15,
    d_model=128,
    n_heads=4,
    n_layers=3,
    n_quantiles=3           # [q_0.1, q_0.5, q_0.9]
)

# 2. Instantiate PyTorch Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PSANet(cfg).to(device)

print(f"PSA-Net Parameter Count: {model.param_count():,}")
```

---

### 2. Training Loop & Pinball (Quantile) Loss

```python
import torch
from psanet.losses import quantile_loss

# Dummy input tensors: [Batch, input_window, N_features]
batch_size = 32
x_hist = torch.randn(batch_size, cfg.input_window, cfg.n_features).to(device)
y_target = torch.randn(batch_size, cfg.forecast_horizon, cfg.n_features).to(device)

# Time indices (optional for seasonal embeddings)
n_patches = (cfg.input_window - cfg.patch_len) // cfg.patch_stride + 1
tod_idx = torch.randint(0, cfg.steps_per_day, (batch_size, n_patches)).to(device)
dow_idx = torch.randint(0, cfg.days_per_week, (batch_size, n_patches)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training Step
model.train()
optimizer.zero_grad()

# Forward Pass -> Output shape: [batch_size, forecast_horizon, n_features, n_quantiles]
preds = model(x_hist, tod_idx, dow_idx)

# Compute Quantile Pinball Loss for q=[0.1, 0.5, 0.9]
loss = quantile_loss(preds, y_target, quantiles=[0.1, 0.5, 0.9])
loss.backward()
optimizer.step()

print(f"Training Step Loss: {loss.item():.4f}")
```

---

### 3. Inference & Quantile Unscaling

```python
import torch
import numpy as np

# Set model to evaluation mode
model.eval()

# Assume mean and std are normalization statistics fit on training data
mean = np.zeros(cfg.n_features)
std = np.ones(cfg.n_features)

# Historical context window: [1, input_window, n_features]
context_raw = np.random.randn(cfg.input_window, cfg.n_features)
context_norm = (context_raw - mean) / std
context_tensor = torch.tensor(context_norm, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    raw_preds = model(context_tensor)  # [1, forecast_horizon, n_features, 3]

# Unscale back to original metric units
preds_np = raw_preds.squeeze(0).cpu().numpy()  # [forecast_horizon, n_features, 3]
preds_unscaled = preds_np * std[None, :, None] + mean[None, :, None]

# Extract Quantile Curves
q_10 = preds_unscaled[..., 0]  # Lower Bound (q=0.1)
q_50 = preds_unscaled[..., 1]  # Median Point Forecast (q=0.5)
q_90 = preds_unscaled[..., 2]  # Upper Bound (q=0.9)
```

---

### 4. Saving and Loading Checkpoints

```python
import torch

# Save Checkpoint
checkpoint_dict = {
    "model_state": model.state_dict(),
    "config": cfg,
    "mean": mean,
    "std": std
}
torch.save(checkpoint_dict, "psanet_checkpoint.pt")

# Load Checkpoint
loaded_ckpt = torch.load("psanet_checkpoint.pt", map_location="cpu", weights_only=False)
loaded_cfg = loaded_ckpt["config"]

loaded_model = PSANet(loaded_cfg)
loaded_model.load_state_dict(loaded_ckpt["model_state"])
loaded_model.eval()
```

---

## Limitations

1. **Minimum History Requirement**: Requires at least `input_window` continuous historical timesteps before generating forecasts.
2. **Channel-Mixing Parameter Scaling**: Because the output head flattens all features and patches into a single linear projection, parameter count scales with $N_{\text{features}} \times N_{\text{patches}} \times \text{d\_model}$. For setups with $>20$ features, consider using a channel-independent projection head.

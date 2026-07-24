"""
PSA-Net — Patch-Spectral Attention Network
============================================

A configurable, dataset-agnostic architecture for multivariate time-series
forecasting, combining three ingredients (each individually well-understood
and known to train reliably) in a specific arrangement targeted at
telemetry-style data with strong seasonality and spike events:

  1. Patch tokenization (PatchTST-style): each feature's history is chunked
     into overlapping patches before attention, instead of per-timestep
     tokens. Cheaper, and patches capture local shape (e.g. the leading
     edge of a spike) directly.

  2. Dual time/spectral branches (TSPulse-inspired): each patch is encoded
     both in the raw time domain and via FFT into the frequency domain,
     then fused. Daily/weekly seasonality shows up as strong low-frequency
     components, giving seasonality a dedicated representational path
     instead of relying on attention alone to discover it from raw position.

  3. Spike-pattern cross-attention: a learned bank of reference "spike
     shape" embeddings that patches attend to explicitly, giving the model
     a direct mechanism to recognize "this looks like the start of a spike
     we've seen before," rather than only reacting to a raw magnitude
     threshold.

No custom low-level math: patching, standard multi-head attention, and FFT
are all standard, well-documented primitives. The novelty is in the
arrangement and in targeting it specifically at multivariate telemetry
forecasting with known seasonality/spike structure.

Attention-only — no SSM/recurrence. Long-range context (daily/weekly
seasonality) is handled by the spectral branch + explicit seasonal
embeddings, not by a scan/recurrence mechanism.
"""

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PSANetConfig:
    # --- data shape (dataset-agnostic; set per dataset) ---
    n_features: int = 6          # number of input signals (works for 3, 6, 8, 20, ...)
    input_window: int = 1440     # history length in timesteps (e.g. 1440 = 24h @ 1min, 288 = 24h @ 5min)
    forecast_horizon: int = 60   # steps to predict (e.g. 60 = 1h @ 1min, 12 = 1h @ 5min)

    # --- patching ---
    patch_len: int = 60          # timesteps per patch (e.g. 60 = 1h @ 1min, 12 = 1h @ 5min)
    patch_stride: int = 30       # stride between patches (overlap if < patch_len)

    # --- model size ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3            # number of patch-attention blocks
    d_ff: int = 256              # feedforward hidden dim inside each block
    dropout: float = 0.1

    # --- spectral branch ---
    use_spectral_branch: bool = True
    n_freq_bins: int = 32        # number of low-frequency FFT bins kept per patch

    # --- seasonal embeddings ---
    use_seasonal_embed: bool = True
    steps_per_day: int = 1440    # for time-of-day embedding: 1440 @ 1min resolution, 288 @ 5min. MUST match data.
    days_per_week: int = 7

    # --- spike pattern bank ---
    use_spike_bank: bool = True
    n_spike_patterns: int = 16   # number of learned reference spike-shape embeddings
    spike_bank_heads: int = 4

    # --- uncertainty ---
    n_quantiles: int = 3         # e.g. [0.1, 0.5, 0.9]; set to 1 for point forecast only

    # --- fallback path (see README) ---
    # If the full model is unstable/slow to converge under deadline pressure,
    # set use_spectral_branch=False and use_spike_bank=False to fall back to
    # a plain patch-attention forecaster (PatchTST-equivalent) with almost no
    # code changes required.


class PatchEmbed(nn.Module):
    """Splits [B, T, F] into overlapping patches per feature and embeds them."""

    def __init__(self, cfg: PSANetConfig):
        super().__init__()
        self.cfg = cfg
        self.n_patches = (cfg.input_window - cfg.patch_len) // cfg.patch_stride + 1
        # each patch is patch_len raw values (per feature); project to d_model
        self.time_proj = nn.Linear(cfg.patch_len, cfg.d_model)
        self.feature_embed = nn.Embedding(cfg.n_features, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, Fdim = x.shape
        cfg = self.cfg
        # unfold time dim into patches: [B, F, n_patches, patch_len]
        x = x.permute(0, 2, 1)  # [B, F, T]
        patches = x.unfold(dimension=2, size=cfg.patch_len, step=cfg.patch_stride)
        # patches: [B, F, n_patches, patch_len]
        n_patches = patches.shape[2]

        time_emb = self.time_proj(patches)  # [B, F, n_patches, d_model]

        feat_ids = torch.arange(Fdim, device=x.device)
        feat_emb = self.feature_embed(feat_ids).view(1, Fdim, 1, -1)  # [1, F, 1, d_model]
        time_emb = time_emb + feat_emb

        # flatten (feature, patch) into one token sequence per batch item
        tokens = time_emb.reshape(B, Fdim * n_patches, cfg.d_model)
        return tokens, patches, n_patches  # also return raw patches for spectral branch


class SpectralBranch(nn.Module):
    """FFT-based frequency-domain encoding of each patch, fused with the time-domain token."""

    def __init__(self, cfg: PSANetConfig):
        super().__init__()
        self.cfg = cfg
        # rfft of a length-patch_len signal gives patch_len//2 + 1 complex bins;
        # keep only the lowest n_freq_bins (captures seasonality-relevant low frequencies)
        self.n_bins = min(cfg.n_freq_bins, cfg.patch_len // 2 + 1)
        self.freq_proj = nn.Linear(self.n_bins * 2, cfg.d_model)  # *2 for magnitude+phase
        self.fuse = nn.Linear(cfg.d_model * 2, cfg.d_model)

    def forward(self, time_tokens: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, F, n_patches, patch_len]
        spec = torch.fft.rfft(patches, dim=-1)  # [B, F, n_patches, patch_len//2+1] complex
        spec = spec[..., : self.n_bins]
        mag = spec.abs()
        phase = spec.angle()
        spec_feat = torch.cat([mag, phase], dim=-1)  # [B, F, n_patches, n_bins*2]

        B, Fdim, n_patches, _ = spec_feat.shape
        spec_emb = self.freq_proj(spec_feat)  # [B, F, n_patches, d_model]
        spec_emb = spec_emb.reshape(B, Fdim * n_patches, -1)

        fused = self.fuse(torch.cat([time_tokens, spec_emb], dim=-1))
        return fused


class SeasonalEmbed(nn.Module):
    """Time-of-day + day-of-week embeddings, added per timestep-position of each patch's start."""

    def __init__(self, cfg: PSANetConfig):
        super().__init__()
        self.cfg = cfg
        self.tod_embed = nn.Embedding(cfg.steps_per_day, cfg.d_model)
        self.dow_embed = nn.Embedding(cfg.days_per_week, cfg.d_model)

    def forward(self, tokens: torch.Tensor, tod_idx: torch.Tensor, dow_idx: torch.Tensor,
                n_features: int, n_patches: int) -> torch.Tensor:
        # tod_idx, dow_idx: [B, n_patches] — time-of-day / day-of-week at each patch's start
        tod = self.tod_embed(tod_idx)  # [B, n_patches, d_model]
        dow = self.dow_embed(dow_idx)  # [B, n_patches, d_model]
        seasonal = (tod + dow)  # [B, n_patches, d_model]
        # broadcast across features (each feature's patches at same time index share seasonal ctx)
        seasonal = seasonal.unsqueeze(1).expand(-1, n_features, -1, -1)
        seasonal = seasonal.reshape(tokens.shape[0], n_features * n_patches, -1)
        return tokens + seasonal


class SpikePatternBank(nn.Module):
    """Learned reference spike-shape embeddings; tokens cross-attend to this bank."""

    def __init__(self, cfg: PSANetConfig):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(cfg.n_spike_patterns, cfg.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.spike_bank_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        bank = self.bank.unsqueeze(0).expand(B, -1, -1)  # [B, n_patterns, d_model]
        out, _ = self.attn(query=tokens, key=bank, value=bank)
        return tokens + self.norm(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: PSANetConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a))
        f = self.ff(x)
        x = self.norm2(x + self.drop(f))
        return x


class PSANet(nn.Module):
    """
    Full PSA-Net forecaster.

    forward(x, tod_idx, dow_idx) -> quantile forecasts [B, forecast_horizon, n_features, n_quantiles]

    x: [B, input_window, n_features] raw (normalized) history
    tod_idx: [B, n_patches] time-of-day index at each patch's starting timestep
    dow_idx: [B, n_patches] day-of-week index at each patch's starting timestep
    """

    def __init__(self, cfg: PSANetConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.spectral = SpectralBranch(cfg) if cfg.use_spectral_branch else None
        self.seasonal = SeasonalEmbed(cfg) if cfg.use_seasonal_embed else None
        self.spike_bank = SpikePatternBank(cfg) if cfg.use_spike_bank else None
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        n_patches = self.patch_embed.n_patches
        flat_dim = cfg.n_features * n_patches * cfg.d_model
        self.head = nn.Linear(
            flat_dim, cfg.forecast_horizon * cfg.n_features * cfg.n_quantiles
        )

    def forward(self, x: torch.Tensor, tod_idx: torch.Tensor = None,
                dow_idx: torch.Tensor = None) -> torch.Tensor:
        cfg = self.cfg
        tokens, patches, n_patches = self.patch_embed(x)

        if self.spectral is not None:
            tokens = self.spectral(tokens, patches)

        if self.seasonal is not None and tod_idx is not None:
            tokens = self.seasonal(tokens, tod_idx, dow_idx, cfg.n_features, n_patches)

        if self.spike_bank is not None:
            tokens = self.spike_bank(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        B = tokens.shape[0]
        out = self.head(tokens.reshape(B, -1))
        out = out.reshape(B, cfg.forecast_horizon, cfg.n_features, cfg.n_quantiles)
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

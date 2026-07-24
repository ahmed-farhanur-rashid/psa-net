"""
Training script for PSA-Net on multivariate telemetry forecasting.

Expects a CSV with a `timestamp` column plus N numeric feature columns
(e.g. requests_per_second, cpu_utilization_pct, ...).

Usage:
    python -m psanet.train --config configs/base.yaml \
        --csv data/02_processed/hpa_data.csv \
        --features requests_per_second cpu_utilization_pct gpu_utilization_pct

Any field in the YAML config can be overridden from the CLI, e.g.
`--epochs 50 --d_model 256`.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from psanet.model import PSANet, PSANetConfig
from psanet.dataset import make_splits
from psanet.losses import quantile_loss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "models", "psa-net.pt")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_arg_parser(defaults: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--features", nargs="+", required=True,
                    help="Target columns first, then any context-only columns (see --context_features).")
    p.add_argument("--context_features", nargs="+", default=[],
                    help="Feature columns used as model input but NOT forecast (e.g. cluster id, pod_count). "
                         "Appended after --features; excluded from the loss/output.")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience (epochs without val improvement).")

    # every config field is CLI-overridable; type inferred from the YAML default
    for key, val in defaults.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", type=lambda x: x.lower() == "true", default=None)
        elif isinstance(val, list):
            continue  # quantiles list: not exposed as a CLI flag, edit the YAML directly
        else:
            p.add_argument(f"--{key}", type=type(val), default=None)
    return p


def train(args, cfg_dict: dict):
    import pandas as pd

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, csv_path)

    df = pd.read_csv(csv_path)
    n_targets = len(args.features)
    feature_cols = args.features + args.context_features  # targets first, then context-only cols

    cfg = PSANetConfig(
        n_features=len(feature_cols),
        n_targets=n_targets,
        input_window=cfg_dict["input_window"],
        forecast_horizon=cfg_dict["forecast_horizon"],
        steps_per_day=cfg_dict["steps_per_day"],
        patch_len=cfg_dict["patch_len"],
        patch_stride=cfg_dict["patch_stride"],
        d_model=cfg_dict["d_model"],
        n_heads=cfg_dict["n_heads"],
        n_layers=cfg_dict["n_layers"],
        d_ff=cfg_dict["d_ff"],
        dropout=cfg_dict["dropout"],
        use_spectral_branch=cfg_dict["use_spectral_branch"],
        n_freq_bins=cfg_dict["n_freq_bins"],
        use_seasonal_embed=cfg_dict["use_seasonal_embed"],
        days_per_week=cfg_dict["days_per_week"],
        use_spike_bank=cfg_dict["use_spike_bank"],
        n_spike_patterns=cfg_dict["n_spike_patterns"],
        spike_bank_heads=cfg_dict["spike_bank_heads"],
        n_quantiles=cfg_dict["n_quantiles"],
    )

    train_ds, val_ds = make_splits(df, feature_cols, cfg, val_fraction=cfg_dict["val_fraction"])

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=cfg_dict["batch_size"], shuffle=True,
                               drop_last=True, num_workers=num_workers, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=min(cfg_dict["batch_size"], len(val_ds)),
                             shuffle=False, num_workers=num_workers, pin_memory=True,
                             persistent_workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PSANet(cfg).to(device)
    print(f"Model params: {model.param_count():,}  |  device: {device}  |  targets: {n_targets}")

    # GPU optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled (reduce-overhead)")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg_dict["lr"], weight_decay=cfg_dict["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg_dict["epochs"], eta_min=cfg_dict["lr"] * 0.01)
    scaler = GradScaler(enabled=device == "cuda")
    quantiles = cfg_dict["quantiles"]

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    patience = args.patience

    t_start = time.time()

    for epoch in range(cfg_dict["epochs"]):
        model.train()
        train_loss = 0.0
        for hist, tod, dow, target in train_loader:
            hist = hist.to(device, non_blocking=True)
            tod = tod.to(device, non_blocking=True)
            dow = dow.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device == "cuda"):
                pred = model(hist, tod, dow)
                loss = quantile_loss(pred, target, quantiles)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_dict["grad_clip"])
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist, tod, dow, target in val_loader:
                hist = hist.to(device, non_blocking=True)
                tod = tod.to(device, non_blocking=True)
                dow = dow.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with autocast(enabled=device == "cuda"):
                    pred = model(hist, tod, dow)
                    val_loss += quantile_loss(pred, target, quantiles).item()
        val_loss /= len(val_loader)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            mark = " *best*"
        else:
            epochs_no_improve += 1
            mark = ""

        print(f"epoch {epoch+1:3d}/{cfg_dict['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{mark}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - t_start
    print(f"Training complete in {elapsed:.0f}s (best val_loss={best_val_loss:.5f} at epoch {best_epoch})")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(PROJECT_ROOT) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "mean": train_ds.mean, "std": train_ds.std}, out_path)
    print(f"Saved to {out_path}")


def main():
    # first pass: just get --config so we know the YAML defaults before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="configs/base.yaml")
    known, _ = pre.parse_known_args()

    cfg_dict = load_config(known.config)
    parser = build_arg_parser(cfg_dict)
    args = parser.parse_args()

    # apply any CLI overrides on top of the YAML defaults
    for key in cfg_dict:
        override = getattr(args, key, None)
        if override is not None:
            cfg_dict[key] = override

    train(args, cfg_dict)


if __name__ == "__main__":
    main()

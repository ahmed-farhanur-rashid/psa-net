"""Loss functions for PSA-Net."""

import torch


def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """
    Pinball / quantile loss, averaged over quantiles, horizon, features, batch.

    preds: [B, horizon, F, n_quantiles]
    target: [B, horizon, F]
    """
    target = target.unsqueeze(-1)  # [B, horizon, F, 1]
    losses = []
    for i, q in enumerate(quantiles):
        err = target[..., 0] - preds[..., i]
        losses.append(torch.max((q - 1) * err, q * err))
    pinball = torch.stack(losses, dim=-1).mean()

    # Quantile spread penalty: prevent collapse when n_quantiles >= 3
    if len(quantiles) >= 3:
        spread_penalty = 0.0
        for i in range(len(quantiles) - 1):
            gap = (preds[..., i + 1] - preds[..., i]).clamp(min=0)
            spread_penalty += torch.relu(0.05 - gap).mean()
        return pinball + 0.1 * spread_penalty

    return pinball

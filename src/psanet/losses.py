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
    return torch.stack(losses, dim=-1).mean()

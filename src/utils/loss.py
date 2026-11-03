"""Confidence-weighted pose losses."""

import torch
import torch.nn as nn


class WeightedMPJPELoss(nn.Module):
    def forward(self, prediction, target, visibility, source):
        error = torch.linalg.vector_norm(prediction - target, dim=-1)
        weights = visibility.clamp(0.05, 1.0)
        pseudo = torch.tensor([item == "real_pseudo" for item in source],
                              device=prediction.device).unsqueeze(1)
        weights = weights * torch.where(pseudo, 0.5, 1.0)
        return (error * weights).sum() / weights.sum().clamp_min(1.0)


def bone_length_loss(prediction, pairs=((0, 1), (1, 2), (2, 3), (3, 4),
                                        (1, 5), (5, 6), (6, 7))):
    lengths = [torch.linalg.vector_norm(prediction[:, start] - prediction[:, end], dim=-1)
               for start, end in pairs]
    return torch.stack(lengths, dim=1).mean()

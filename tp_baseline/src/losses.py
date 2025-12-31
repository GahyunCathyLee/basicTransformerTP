# tp_baseline/src/losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_loss(name: str = "smoothl1"):
    if name.lower() == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss()

def build_gt_delta(y_abs: torch.Tensor, x_last_abs: torch.Tensor) -> torch.Tensor:
    """
    y_abs: (B,Tf,2)
    x_last_abs: (B,2)
    returns y_delta: (B,Tf,2)
    delta[0] = y[0] - x_last
    delta[t] = y[t] - y[t-1]
    """
    B, Tf, _ = y_abs.shape
    prev = torch.cat([x_last_abs.unsqueeze(1), y_abs[:, :-1, :]], dim=1)  # (B,Tf,2)
    return y_abs - prev

def delta_to_abs(pred_delta: torch.Tensor, x_last_abs: torch.Tensor) -> torch.Tensor:
    """
    pred_delta: (B,Tf,2)
    x_last_abs: (B,2)
    """
    return torch.cumsum(pred_delta, dim=1) + x_last_abs.unsqueeze(1)

def trajectory_loss(
    pred: torch.Tensor,
    y_abs: torch.Tensor,
    x_last_abs: torch.Tensor,
    predict_delta: bool,
    w_traj: float = 1.0,
    w_fde: float = 0.0,
) -> torch.Tensor:
    """
    If predict_delta:
      - loss on delta space (SmoothL1 over all steps)
      - optional last-step loss on absolute (FDE-weighted)
    Else:
      - loss on absolute space
      - optional last-step loss on absolute
    """
    if predict_delta:
        y_delta = build_gt_delta(y_abs, x_last_abs)
        loss_traj = F.smooth_l1_loss(pred, y_delta)
        if w_fde > 0:
            pred_abs = delta_to_abs(pred, x_last_abs)
            loss_last = F.smooth_l1_loss(pred_abs[:, -1, :], y_abs[:, -1, :])
            return w_traj * loss_traj + w_fde * loss_last
        return w_traj * loss_traj

    # absolute prediction
    loss_traj = F.smooth_l1_loss(pred, y_abs)
    if w_fde > 0:
        loss_last = F.smooth_l1_loss(pred[:, -1, :], y_abs[:, -1, :])
        return w_traj * loss_traj + w_fde * loss_last
    return w_traj * loss_traj

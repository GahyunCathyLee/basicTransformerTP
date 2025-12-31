# tp_baseline/src/metrics.py
import torch

def ade(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # (B,Tf,2)
    return torch.norm(y_hat - y, dim=-1).mean()

def fde(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(y_hat[:, -1] - y[:, -1], dim=-1).mean()

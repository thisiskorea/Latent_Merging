"""
CKA and midness utilities for latent mixing evaluation.
"""
from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, Optional


def _normalize(v: Tensor, eps: float = 1e-9) -> Tensor:
    return v / (v.norm(p=2, dim=-1, keepdim=True) + eps)


def _center_rows(X: Tensor) -> Tensor:
    return X - X.mean(dim=0, keepdim=True)


@torch.no_grad()
def linear_cka(X: Tensor, Y: Tensor) -> Optional[float]:
    """
    Centered linear CKA for two sequences (T,H). Returns None on shape mismatch.
    """
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        return None
    Xc, Yc = _center_rows(X), _center_rows(Y)
    XtY = Xc.T @ Yc
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    num = (XtY * XtY).sum()
    den = torch.sqrt((XtX * XtX).sum()) * torch.sqrt((YtY * YtY).sum())
    return (num / (den + 1e-12)).item()


@torch.no_grad()
def _slerp(u: Tensor, v: Tensor, t: float, eps: float = 1e-6) -> Tensor:
    u = _normalize(u, eps)
    v = _normalize(v, eps)
    dot = (u * v).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    small = theta < 1e-3
    sin_t = torch.sin(theta).clamp_min(eps)
    out = (torch.sin((1 - t) * theta) / sin_t) * u + (torch.sin(t * theta) / sin_t) * v
    out_lin = (1 - t) * u + t * v
    return _normalize(torch.where(small, out_lin, out), eps)


@torch.no_grad()
def _geo_dist(u: Tensor, v: Tensor) -> Tensor:
    u = _normalize(u)
    v = _normalize(v)
    return torch.acos((u * v).sum(dim=-1).clamp(-1.0, 1.0))


@torch.no_grad()
def midness_seq(Hb: Tensor, Hf: Tensor, Hm: Tensor) -> Dict[str, float]:
    """
    Hb/Hf/Hm: (T,H) sequences. Returns midpoint cosine, arc deviation, arc ratio.
    """
    def as2d(x: Tensor) -> Tensor:
        return x if x.ndim == 2 else x.unsqueeze(0)

    B, F, M = as2d(Hb), as2d(Hf), as2d(Hm)
    Mid = _slerp(B, F, 0.5)  # (T,H)
    cos_mid = (_normalize(M) * Mid).sum(dim=-1)  # (T,)
    d_bf = _geo_dist(B, F)  # (T,)
    d_bm = _geo_dist(B, M)  # (T,)
    arc_dev = torch.abs(d_bm - 0.5 * d_bf)  # (T,)
    arc_ratio = d_bm / (d_bf + 1e-12)  # (T,)
    return {
        "midpoint_cos": float(cos_mid.mean().item()),
        "arc_mid_deviation": float(arc_dev.mean().item()),
        "arc_ratio": float(arc_ratio.mean().item()),
    }

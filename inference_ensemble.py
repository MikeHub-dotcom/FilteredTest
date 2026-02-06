#!/usr/bin/env python3
"""
Professional inference + trade recommendation script with configurable ensembles.

Key features:
- Load one or multiple checkpoints (TCN / PatchTST / iTransformer via main.build_model).
- Ensemble combination in logit space (default) or probability space; optional weights.
- Robust threshold policies:
    * fixed:      user-provided thr_enter
    * ckpt:       average of checkpoint val_thr (if available)
    * quantile:   choose thr such that top-q of assets are candidates (drift-robust)
- Hysteresis: separate enter/exit thresholds.
- Uses dataset mask explicitly; avoids "0.0 as missing" pitfalls by passing mask to models when supported.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from contextlib import nullcontext

# Backtest plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.nn as nn
import numpy as np
import torch

# Reuse training code (architectures + utils) without duplication.
# Prefer the current training entrypoint (main_train.py). Fallback to main.py for older layouts.
try:
    import main_train as train_mod
except ImportError:
    # main.py is safe to import because it only executes main() under __name__=='__main__'.
    import main as train_mod


# -----------------------------
# Logging
# -----------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger


# -----------------------------
# I/O helpers
# -----------------------------
def load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected .npy, got .npz: {path}")
    return arr


def load_text_lines(path: str) -> List[str]:
    return train_mod.load_text_lines(path)


def load_feature_names(path: str, expected_F: int) -> List[str]:
    """
    Load feature names from either:
      - text file: one name per line
      - json: ["f1", "f2", ...]
      - json nested wrapper: [["f1", ...]]   (your case)
      - json dict: {"features":[...]} or {"feature_names":[...]}
    Enforces exact expected_F.
    """
    p = Path(path)
    suf = p.suffix.lower()

    if suf != ".json":
        feats = [x.strip() for x in load_text_lines(path) if x.strip()]
        if len(feats) != expected_F:
            raise ValueError(
                f"Feature count mismatch (text): expected F={expected_F}, got {len(feats)} from {path}"
            )
        return feats

    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        feats = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("features"), list):
            feats = obj["features"]
        elif isinstance(obj.get("feature_names"), list):
            feats = obj["feature_names"]
        else:
            raise ValueError(f"Unsupported features JSON schema in {path}. Keys={list(obj.keys())}")
    else:
        raise ValueError(f"Invalid JSON in {path}: expected list or dict, got {type(obj)}")

    # Unwrap a single outer list wrapper: [[...]] -> [...]
    if len(feats) == 1 and isinstance(feats[0], list):
        feats = feats[0]

    feats = [str(x).strip() for x in feats if str(x).strip()]
    if len(feats) != expected_F:
        raise ValueError(
            f"Feature count mismatch (json): expected F={expected_F}, got {len(feats)} from {path}. "
            f"First entries: {feats[:10]}"
        )
    return feats


# -----------------------------
# Legacy TCN (old-commit compatibility)
# -----------------------------
class _LegacyResidualTCNBlock(nn.Module):
    """
    Matches old commit: two dilated convs + (GroupNorm or BatchNorm) + GELU + Dropout + residual proj.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_groupnorm: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)

        if use_groupnorm:
            g1 = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
            self.norm1 = nn.GroupNorm(g1, out_ch)
            self.norm2 = nn.GroupNorm(g1, out_ch)
        else:
            self.norm1 = nn.BatchNorm1d(out_ch)
            self.norm2 = nn.BatchNorm1d(out_ch)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.drop(h)

        r = x if self.proj is None else self.proj(x)
        return h + r


class LegacyTCNModel(nn.Module):
    """
    Exact head naming from old commit:
      backbone.*, pool.*, head.1.*, head.4.*
    Input: (B, C, L)
    Output: (B, 1) logits
    """
    def __init__(
        self,
        in_ch: int,
        channels: int = 192,
        levels: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.10,
        use_groupnorm: bool = True,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        ch_in = int(in_ch)
        for i in range(int(levels)):
            d = 2 ** i
            blocks.append(
                _LegacyResidualTCNBlock(
                    in_ch=ch_in,
                    out_ch=int(channels),
                    kernel_size=int(kernel_size),
                    dilation=int(d),
                    dropout=float(dropout),
                    use_groupnorm=bool(use_groupnorm),
                )
            )
            ch_in = int(channels)

        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(channels), int(channels) // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(channels) // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.pool(z)
        return self.head(z)


def _ckpt_looks_like_legacy_tcn(sd: Dict[str, torch.Tensor]) -> bool:
    # legacy has head.* keys; new refactor tends to have pre./out. keys
    has_head = any(k.startswith("head.") for k in sd.keys())
    has_pre_or_out = any(k.startswith("pre.") or k.startswith("out.") for k in sd.keys())
    return bool(has_head and not has_pre_or_out)


# -----------------------------
# Scoring helpers (ranking)
# -----------------------------
def compute_recent_vol(close: np.ndarray, lookback: int = 20) -> float:
    """
    close: [L] close prices for a single asset
    returns realized vol of log returns, annualization not required (used for ranking only)
    """
    close = np.asarray(close, dtype=np.float64)
    if close.size < 3:
        return float("nan")
    lb = min(int(lookback), close.size - 1)
    x = close[-(lb + 1):]
    r = np.diff(np.log(np.maximum(x, 1e-12)))
    if r.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(r * r)))


def rank_score(p: float, thr_enter: float, vol: float, mode: str) -> float:
    """
    Convert probability to a ranking score.
    - p: predicted prob of "up" / buy
    - thr_enter: entry threshold used as reference
    - vol: recent realized vol (for margin_over_vol)
    """
    if not np.isfinite(p):
        return -1e9
    margin = p - float(thr_enter)
    if mode == "p":
        return float(p)
    if mode == "margin":
        return float(margin)
    if mode == "margin_over_vol":
        denom = max(float(vol), 1e-6) if np.isfinite(vol) else 1e-6
        return float(margin / denom)
    raise ValueError(f"unknown rank_mode: {mode}")


# -----------------------------
# Model loading / compatibility
# -----------------------------
def load_checkpoint(path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)
    if "model_state" not in ckpt:
        raise RuntimeError(f"Checkpoint missing 'model_state': {path}")
    return ckpt



def _infer_model_type_from_state_dict(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Best-effort inference for legacy checkpoints that don't store cfg.model_type.
    """
    keys = set(sd.keys())
    if "proj.weight" in keys and "pos" in keys and any(k.startswith("encoder.layers.") for k in keys):
        return "patchtst"
    if any(".backbone." in k for k in keys) and any(".head." in k for k in keys):
        # tcn/cnn share head naming in the legacy script; keep None if ambiguous
        return None
    return None


def _patchtst_cfg_from_state_dict(cfg: train_mod.Config, sd: Dict[str, torch.Tensor], *, in_ch: int) -> train_mod.Config:
    """
    PatchTST proj/pos shapes depend on (in_ch, patch_len, patch_stride, lookback, d_model).
    Infer the necessary values from the checkpoint weights to prevent size-mismatch on load.
    """
    cfg2 = copy.deepcopy(cfg)

    # proj.weight: (d_model, token_dim) with token_dim = in_ch * patch_len
    if "proj.weight" in sd:
        d_model, token_dim = map(int, sd["proj.weight"].shape)
        if hasattr(cfg2, "d_model"):
            cfg2.d_model = int(d_model)
        if token_dim % int(in_ch) == 0:
            cfg2.patch_len = int(token_dim // int(in_ch))

    # pos: (1, 1+n_patches, d_model) with n_patches derived from lookback/stride/patch_len
    if "pos" in sd and hasattr(cfg2, "patch_len") and hasattr(cfg2, "patch_stride"):
        n_patches = int(sd["pos"].shape[1]) - 1
        # force build-time n_patches to match ckpt by choosing a consistent lookback
        cfg2.lookback = int(cfg2.patch_len + max(0, n_patches - 1) * int(cfg2.patch_stride))

    return cfg2


def load_state_dict_compat(model: torch.nn.Module, sd: Dict[str, torch.Tensor], *, strict: bool = True) -> None:
    """
    Strict-load when possible; otherwise apply minimal, *deterministic* fixes for the common
    PatchTST mismatches (proj/pos) by trunc/pad-copy and retry.
    """
    try:
        model.load_state_dict(sd, strict=strict)
        return
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch" not in msg:
            raise

    sd2 = dict(sd)  # shallow copy is fine (tensors are immutable here)

    # proj.weight: (d_model, token_dim)
    if "proj.weight" in sd2 and hasattr(model, "proj") and isinstance(getattr(model, "proj"), torch.nn.Linear):
        w = sd2["proj.weight"]
        w_tgt = model.proj.weight
        if w.shape != w_tgt.shape:
            d0 = min(w.shape[0], w_tgt.shape[0])
            d1 = min(w.shape[1], w_tgt.shape[1])
            w_new = w_tgt.detach().clone().zero_()
            w_new[:d0, :d1] = w[:d0, :d1]
            sd2["proj.weight"] = w_new
            if "proj.bias" in sd2 and sd2["proj.bias"].shape != model.proj.bias.shape:
                b = sd2["proj.bias"]
                b_new = model.proj.bias.detach().clone().zero_()
                d = min(b.shape[0], b_new.shape[0])
                b_new[:d] = b[:d]
                sd2["proj.bias"] = b_new

    # pos: (1, 1+n_patches, d_model)
    if "pos" in sd2 and hasattr(model, "pos") and isinstance(getattr(model, "pos"), torch.nn.Parameter):
        p = sd2["pos"]
        p_tgt = model.pos
        if p.shape != p_tgt.shape:
            a = min(p.shape[1], p_tgt.shape[1])
            b = min(p.shape[2], p_tgt.shape[2])
            p_new = p_tgt.detach().clone().zero_()
            p_new[:, :a, :b] = p[:, :a, :b]
            sd2["pos"] = p_new

    model.load_state_dict(sd2, strict=False)


def build_model_from_ckpt(ckpt: Dict, in_ch: int, device: torch.device) -> Tuple[torch.nn.Module, train_mod.Config, Optional[float]]:
    if "cfg" not in ckpt:
        raise RuntimeError("Checkpoint has no cfg – cannot reconstruct model safely.")

    cfg = train_mod.Config(**ckpt["cfg"])

    # infer correct in_ch from checkpoint weights
    sd = ckpt["model_state"]

    if cfg.model_type == "tcn":
        # backbone.0.conv1.weight shape = [C_out, C_in, K]
        in_ch = sd["backbone.0.conv1.weight"].shape[1]

    elif cfg.model_type == "patchtst":
        # proj.weight shape = [d_model, token_dim]
        token_dim = sd["proj.weight"].shape[1]
        in_ch = token_dim // cfg.patch_len

    elif cfg.model_type in ("itransformer", "itr", "i_transformer"):
        in_ch = sd["var_proj.weight"].shape[1]

    else:
        raise RuntimeError(f"Unknown model_type={cfg.model_type}")

    model = train_mod.build_model(cfg, in_ch=in_ch).to(device)

    load_state_dict_compat(model, sd)

    model.eval()
    thr = ckpt.get("val_thr")
    thr_f = float(thr) if thr is not None else None

    return model, cfg, thr_f


def _forward_logits(model: torch.nn.Module, xb: torch.Tensor, mb: torch.Tensor) -> torch.Tensor:
    """
    Compatibility shim:
    - Some models expect (xb, mb) where xb=[B,L,F], mb=[B,L,F]
    - Some (e.g., TCN Conv1d) expect concatenated [B,2F,L] only.
    """

    def _infer_expected_in_ch(m: torch.nn.Module) -> Optional[int]:
        # Prefer explicit attribute (used by PatchTST/iTransformer in your training code)
        if hasattr(m, "in_ch"):
            try:
                return int(getattr(m, "in_ch"))
            except Exception:
                pass
        # Fallback: first Conv1d encountered
        for mm in m.modules():
            if isinstance(mm, nn.Conv1d):
                return int(mm.in_channels)
        return None

    def _as_bcl(x_lf: torch.Tensor) -> torch.Tensor:
        # [B,L,F] -> [B,F,L]
        return x_lf.permute(0, 2, 1).contiguous()

    exp_in = _infer_expected_in_ch(model)
    b, l, f = xb.shape

    # Decide whether model expects features-only (F) or features+mask (2F)
    wants_2f = (exp_in == 2 * f)
    wants_f = (exp_in == f)

    # Prepare candidates (most common conventions)
    x_f_bcl = _as_bcl(xb)  # [B,F,L]
    m_f_bcl = _as_bcl(mb)  # [B,F,L]
    x_2f_bcl = _as_bcl(torch.cat([xb, mb], dim=-1))  # [B,2F,L]

    # 1) If model clearly expects F (PatchTST/iTransformer trained on features only)
    if wants_f:
        # Try (xb,mb) first (some legacy codepaths accept two args)
        try:
            out = model(x_f_bcl, m_f_bcl)  # type: ignore[misc]
            return out.view(-1)
        except TypeError:
            out = model(x_f_bcl)  # type: ignore[misc]
            return out.view(-1)
        except RuntimeError:
            # Fallback: try raw layout if forward was written for [B,L,F]
            try:
                out = model(xb, mb)  # type: ignore[misc]
                return out.view(-1)
            except TypeError:
                out = model(xb)  # type: ignore[misc]
                return out.view(-1)

    # 2) If model clearly expects 2F (TCN/CNN that consumes explicit mask)
    if wants_2f:
        try:
            out = model(x_2f_bcl)  # type: ignore[misc]
            return out.view(-1)
        except TypeError:
            out = model(x_2f_bcl, None)  # type: ignore[misc]
            return out.view(-1)

    # 3) Unknown: try a robust cascade
    #    (a) as-is two-arg
    try:
        out = model(xb, mb)  # type: ignore[misc]
        return out.view(-1)
    except TypeError:
        pass
    except RuntimeError:
        # Try transposed two-arg
        try:
            out = model(x_f_bcl, m_f_bcl)  # type: ignore[misc]
            return out.view(-1)
        except Exception:
            pass

    #    (b) single-arg with concat+transpose (legacy TCN path)
    out = model(x_2f_bcl)  # type: ignore[misc]
    return out.view(-1)


@torch.no_grad()
def infer_probs_for_date(
    models: List[torch.nn.Module],
    Xn: np.ndarray,
    M: np.ndarray,
    *,
    t_idx: int,
    lookback: int,
    batch_assets: int,
    device: torch.device,
    amp: bool = True,
    ensemble_mode: str = "logit_mean",
    ensemble_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Compute per-asset probabilities for one time index.

    Xn: standardized cube [A,T,F]
    M:  mask cube [A,T,F] uint8/bool (1=observed)
    """
    assert lookback >= 1
    A, T, F = Xn.shape
    if not (0 <= t_idx < T):
        raise ValueError(f"t_idx out of range: {t_idx} for T={T}")
    if t_idx - lookback + 1 < 0:
        raise ValueError(f"t_idx={t_idx} too early for lookback={lookback}")

    # weights
    w: Optional[np.ndarray]
    if ensemble_weights is not None:
        if len(ensemble_weights) != len(models):
            raise ValueError(f"--ensemble_weights length {len(ensemble_weights)} != n_models {len(models)}")
        w = np.asarray(ensemble_weights, dtype=np.float64)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("--ensemble_weights must be finite and sum to > 0")
        w = w / w.sum()
    else:
        w = None

    window_slice = slice(t_idx - lookback + 1, t_idx + 1)
    Xw = Xn[:, window_slice, :].astype(np.float32, copy=False)  # [A,L,F]
    Mw = M[:, window_slice, :].astype(np.float32, copy=False)   # [A,L,F]

    has_any_obs = (Mw.sum(axis=(1, 2)) > 0)  # [A]
    probs = np.full((A,), np.nan, dtype=np.float32)
    idxs = np.where(has_any_obs)[0]
    if idxs.size == 0:
        return probs

    use_amp = bool(amp and device.type == "cuda")
    try:
        # PyTorch >= 2.0
        from torch.amp import autocast as amp_autocast  # type: ignore

        def _ac(enabled: bool):
            return amp_autocast(device_type=device.type, enabled=enabled)
    except Exception:
        # Older PyTorch fallback
        from torch.cuda.amp import autocast as cuda_autocast  # type: ignore

        def _ac(enabled: bool):
            if device.type == "cuda":
                return cuda_autocast(enabled=enabled)
            return nullcontext()

    for i0 in range(0, idxs.size, batch_assets):
        sub = idxs[i0:i0 + batch_assets]
        xb = torch.from_numpy(Xw[sub]).to(device)
        mb = torch.from_numpy(Mw[sub]).to(device)

        with _ac(enabled=use_amp):
            if ensemble_mode in ("logit_mean", "logit_weighted"):
                logits_list = []
                for mdl in models:
                    logits_list.append(_forward_logits(mdl, xb, mb).float())
                L = torch.stack(logits_list, dim=0)  # [M,B]
                if ensemble_mode == "logit_weighted":
                    if w is None:
                        raise ValueError("ensemble_mode=logit_weighted requires --ensemble_weights")
                    wt = torch.tensor(w, device=device, dtype=L.dtype).view(-1, 1)
                    logits = (wt * L).sum(dim=0)
                else:
                    logits = L.mean(dim=0)
                pb = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

            elif ensemble_mode in ("prob_mean", "prob_median"):
                p_list = []
                for mdl in models:
                    p_list.append(torch.sigmoid(_forward_logits(mdl, xb, mb).float()))
                P = torch.stack(p_list, dim=0)  # [M,B]
                if ensemble_mode == "prob_median":
                    pb = torch.median(P, dim=0).values.detach().cpu().numpy().astype(np.float32)
                else:
                    if w is not None:
                        wt = torch.tensor(w, device=device, dtype=P.dtype).view(-1, 1)
                        pb = (wt * P).sum(dim=0).detach().cpu().numpy().astype(np.float32)
                    else:
                        pb = P.mean(dim=0).detach().cpu().numpy().astype(np.float32)
            else:
                raise ValueError(f"unknown ensemble_mode: {ensemble_mode}")

        probs[sub] = pb

    return probs


# -----------------------------
# Threshold policies
# -----------------------------
def choose_thr_enter(
    probs: np.ndarray,
    *,
    policy: str,
    fixed_thr: Optional[float],
    ckpt_thrs: List[Optional[float]],
    quantile_q: float,
) -> float:
    """
    policy:
      - fixed: use fixed_thr (required)
      - ckpt:  mean of available ckpt val_thr (fallback: 0.5)
      - quantile: choose thr so that top-q assets are candidates (q in (0,1))
    """
    policy = policy.lower()
    if policy == "fixed":
        if fixed_thr is None:
            raise ValueError("--thr_policy fixed requires --thr_enter")
        return float(fixed_thr)

    if policy == "ckpt":
        vals = [t for t in ckpt_thrs if t is not None and math.isfinite(float(t))]
        return float(np.mean(vals)) if vals else 0.5

    if policy == "quantile":
        q = float(quantile_q)
        if not (0 < q < 1):
            raise ValueError("--thr_quantile must be in (0,1)")
        p = probs[np.isfinite(probs)]
        if p.size == 0:
            return 1.0
        # threshold such that approx q fraction are >= thr
        thr = float(np.quantile(p, 1.0 - q))
        return thr

    raise ValueError(f"unknown thr_policy: {policy}")


# -----------------------------
# Portfolio state (simple)
# -----------------------------
def load_positions(path: str) -> Dict:
    if not Path(path).exists():
        return {"positions": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Main inference
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference + trade recommendations (supports ensembles)")

    p.add_argument("--dataset", required=True, help="X cube .npy [A,T,F]")
    p.add_argument("--mask", required=True, help="mask cube .npy [A,T,F] (1=observed)")
    p.add_argument("--assets", required=True, help="assets.txt (A lines)")
    p.add_argument("--dates", required=True, help="dates.txt (T lines, YYYY-MM-DD)")
    p.add_argument("--features", required=True, help="features.txt (F lines) OR features.json ([..] or [[..]])")

    # checkpoints
    mg = p.add_mutually_exclusive_group(required=True)
    mg.add_argument("--models", default=None, help="Comma-separated checkpoint paths (best_model.pt).")
    mg.add_argument("--model", default=None, help="Single checkpoint path (alias for --models).")

    p.add_argument("--ensemble_mode", type=str, default="logit_mean",
                   choices=["logit_mean", "prob_mean", "prob_median", "logit_weighted"])
    p.add_argument("--ensemble_weights", type=str, default="",
                   help='Comma-separated weights (same length as models). Used for logit_weighted and weighted prob_mean.')

    # as-of selection
    g = p.add_mutually_exclusive_group()
    g.add_argument("--asof_date", default=None, help="YYYY-MM-DD present in dates file")
    g.add_argument("--asof_idx", type=int, default=None, help="0-based index into dates")

    # split params for normalization
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)

    # thresholds
    p.add_argument("--thr_policy", choices=["fixed", "ckpt", "quantile"], default="quantile",
                   help="How to choose entry threshold (drift-robust default: quantile).")
    p.add_argument("--thr_enter", type=float, default=None, help="Used when thr_policy=fixed.")
    p.add_argument("--thr_quantile", type=float, default=0.02,
                   help="When thr_policy=quantile: fraction of assets to consider (top-q).")

    p.add_argument("--thr_exit", type=float, default=0.65)
    p.add_argument("--max_positions", type=int, default=8)
    p.add_argument("--max_new_per_day", type=int, default=2)
    p.add_argument("--min_hold_days", type=int, default=3)
    p.add_argument("--cooldown_days", type=int, default=5)
    p.add_argument("--rank_mode", choices=["p", "margin", "margin_over_vol"], default="margin_over_vol")

    p.add_argument("--positions", default="positions.json", help="JSON state file to read.")
    p.add_argument("--outdir", default="runs_inference_ensemble")
    p.add_argument("--write_positions", action="store_true",
                   help="If set, overwrite --positions with updated state.")

    # runtime
    p.add_argument("--device", default=None, help="cuda or cpu; default: cuda if available")
    p.add_argument("--batch_assets", type=int, default=512)
    p.add_argument("--no_amp", action="store_true")

    return p.parse_args()


def main() -> None:
    logger = setup_logger()
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"device={device}")

    # Load data
    X = load_npy(args.dataset)  # [A,T,F]
    M = load_npy(args.mask)
    if M.dtype != np.uint8 and M.dtype != np.bool_:
        M = (M > 0).astype(np.uint8)

    assets = load_text_lines(args.assets)
    dates = load_text_lines(args.dates)

    A, T, F = X.shape

    features = load_feature_names(args.features, expected_F=F)

    # Better diagnostics than a bare assert
    if len(assets) != A or len(dates) != T or len(features) != F:
        raise ValueError(
            "metadata shape mismatch:\n"
            f"  X shape: A={A}, T={T}, F={F}\n"
            f"  assets:   {len(assets)} (file={args.assets})\n"
            f"  dates:    {len(dates)} (file={args.dates})\n"
            f"  features: {len(features)} (file={args.features})\n"
            "Hint: features.json may be nested as [[...]] or from a different export."
        )

    # Choose as-of index
    if args.asof_idx is not None:
        t_idx = int(args.asof_idx)
    elif args.asof_date is not None:
        if args.asof_date not in dates:
            raise ValueError(f"--asof_date {args.asof_date} not found in dates")
        t_idx = dates.index(args.asof_date)
    else:
        t_idx = T - 1
    asof_date = dates[t_idx]
    logger.info(f"asof={asof_date} (t_idx={t_idx})")

    # Normalization (match training)
    # Split indices for calibration window on the full history (up to as-of).
    # We use the same split logic as training code for consistency.
    splits = train_mod.time_split_indices(
        T,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
    )

    # Training code (main_train.py) returns:
    #   (train_start, train_end), (val_start, val_end), (test_start, test_end)
    # where ends are inclusive.
    #
    # Older code may return just (train_end, val_end) or similar.
    if isinstance(splits, tuple) and len(splits) >= 2 and isinstance(splits[0], tuple):
        (tr_s, tr_e) = splits[0]
        (va_s, va_e) = splits[1]
        train_slice = slice(int(tr_s), int(tr_e) + 1)
        cal_slice = slice(int(va_s), int(va_e) + 1)
        # Maintain legacy variables used later in the script (exclusive ends)
        train_end = int(train_slice.stop)
        val_end = int(cal_slice.stop)
    else:
        # Fallback: assume two integers are returned: train_end, val_end (exclusive or inclusive ambiguous)
        # Interpret as inclusive to match the current training convention.
        train_end = int(splits[0])
        val_end = int(splits[1])
        train_slice = slice(0, train_end + 1)
        cal_slice = slice(train_end + 1, val_end + 1)
        # Convert to exclusive ends to match numpy slicing convention in the rest of the code
        train_end = int(train_slice.stop)
        val_end = int(cal_slice.stop)
    mu, sig = train_mod.masked_mean_std(X[:, train_slice, :], M[:, train_slice, :])
    Xn = train_mod.apply_masked_standardize(X, M, mu, sig)

    # checkpoints
    model_list = args.models or args.model
    if model_list is None:
        raise ValueError("must provide --models or --model")
    ckpt_paths = [p.strip() for p in model_list.split(",") if p.strip()]
    if not ckpt_paths:
        raise ValueError("no checkpoints found in --models/--model")

    ensemble_weights = None
    if args.ensemble_weights.strip():
        ensemble_weights = [float(x) for x in args.ensemble_weights.split(",")]

    # Determine in_ch for build_model
    models: List[torch.nn.Module] = []
    ckpt_thrs: List[Optional[float]] = []
    cfgs: List[train_mod.Config] = []


    def _infer_in_ch_from_ckpt(sd: Dict[str, torch.Tensor], cfg: train_mod.Config, model_type: str, F: int) -> Optional[int]:
        """
        Infer required in_ch from checkpoint tensors to avoid ABI mismatches:
        - PatchTST: proj.weight shape is [d_model, in_ch * patch_len]
        - iTransformer: pos shape is [1, tok_count, d_model] where tok_count = in_ch (+1 if CLS)
        Returns inferred in_ch or None if not inferable.
        """
        mt = str(model_type).lower()

        # PatchTST: proj.weight determines token_dim = in_ch * patch_len
        if "patchtst" in mt:
            pw = sd.get("proj.weight", None)
            patch_len = int(getattr(cfg, "patch_len", 0))
            if pw is not None and patch_len > 0:
                token_dim = int(pw.shape[1])
                if token_dim % patch_len == 0:
                    in_ch = token_dim // patch_len
                    # sanity: should typically be F or 2F
                    if in_ch in (int(F), int(2 * F)):
                        return int(in_ch)
                    # still return if plausible (>0)
                    if in_ch > 0:
                        return int(in_ch)

        # iTransformer: pos length reveals token count = in_ch (+1 if CLS)
        if "itransformer" in mt or mt in ("itr", "i_transformer"):
            pos = sd.get("pos", None)
            if pos is not None and pos.ndim == 3:
                tok_count = int(pos.shape[1])
                use_cls = bool(getattr(cfg, "itr_use_cls", True))
                in_ch = tok_count - 1 if use_cls else tok_count
                if in_ch > 0:
                    return int(in_ch)

        return None


    for cp in ckpt_paths:
        ckpt = load_checkpoint(cp, device=device)

        # build cfg first (without building model yet)
        cfg = train_mod.Config()
        if isinstance(ckpt.get("cfg"), dict):
            for k, v in ckpt["cfg"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        # Decide in_ch (checkpoint-driven, ABI-safe):
        model_type = str(getattr(cfg, "model_type", getattr(cfg, "arch", "tcn"))).lower()

        # 1) For TCN/CNN in old pipeline: always concat([X,M]) => 2F
        if "tcn" in model_type or "cnn" in model_type:
            in_ch = int(2 * F)
        else:
            # 2) For transformers: infer from checkpoint if possible (PatchTST/iTransformer),
            #    otherwise default to F.
            inferred = _infer_in_ch_from_ckpt(sd, cfg, model_type=model_type, F=F)
            in_ch = int(inferred) if inferred is not None else int(F)

        sd = ckpt["model_state"]

        # ---- Legacy TCN detection (old training commit) ----
        def _looks_like_legacy_tcn(sd):
            has_head = any(k.startswith("head.") for k in sd.keys())
            has_pre_or_out = any(k.startswith("pre.") or k.startswith("out.") for k in sd.keys())
            return has_head and not has_pre_or_out

        if ("tcn" in model_type) and _looks_like_legacy_tcn(sd):
            mdl = LegacyTCNModel(
                in_ch=in_ch,
                channels=int(getattr(cfg, "tcn_channels", 192)),
                levels=int(getattr(cfg, "tcn_levels", 5)),
                kernel_size=int(getattr(cfg, "tcn_kernel_size", 5)),
                dropout=float(getattr(cfg, "tcn_dropout", getattr(cfg, "dropout", 0.10))),
                use_groupnorm=bool(getattr(cfg, "tcn_use_groupnorm", True)),
            ).to(device)

            logger.info("loaded legacy TCN checkpoint")
        else:
            mdl = train_mod.build_model(cfg, in_ch=in_ch).to(device)

        mdl.eval()

        thr = ckpt.get("val_thr")
        thr_f = float(thr) if thr is not None else None

        models.append(mdl)
        cfgs.append(cfg)
        ckpt_thrs.append(thr_f)
        logger.info(f"loaded {cp} (val_thr={thr_f}, in_ch={in_ch}, model_type={model_type})")

    # Lookback: must be consistent across ensemble; take max to be safe.
    lookbacks = []
    for cfg in cfgs:
        lb = getattr(cfg, "lookback", None)
        if lb is not None:
            lookbacks.append(int(lb))
    lookback = max(lookbacks) if lookbacks else 192
    logger.info(f"ensemble lookback={lookback}")

    # Inference
    probs = infer_probs_for_date(
        models=models,
        Xn=Xn,
        M=M,
        t_idx=t_idx,
        lookback=lookback,
        batch_assets=int(args.batch_assets),
        device=device,
        amp=not bool(args.no_amp),
        ensemble_mode=args.ensemble_mode,
        ensemble_weights=ensemble_weights,
    )

    thr_enter = choose_thr_enter(
        probs,
        policy=str(args.thr_policy),
        fixed_thr=args.thr_enter,
        ckpt_thrs=ckpt_thrs,
        quantile_q=float(args.thr_quantile),
    )
    thr_exit = float(args.thr_exit)
    logger.info(f"thr_enter={thr_enter:.4f} (policy={args.thr_policy}), thr_exit={thr_exit:.4f}")

    # Portfolio simulation for one day: decide BUY/HOLD/SELL
    state = load_positions(args.positions)
    pos: Dict[str, Dict] = state.get("positions", {})
    today_idx = t_idx

    # update existing positions: apply exit + min_hold + cooldown bookkeeping
    actions_sell = []
    actions_hold = []
    updated_pos: Dict[str, Dict] = {}

    for sym, info in pos.items():
        entry_idx = int(info.get("entry_idx", today_idx))
        last_action_idx = int(info.get("last_action_idx", entry_idx))
        min_hold_ok = (today_idx - entry_idx) >= int(args.min_hold_days)

        # cooldown isn't applied to currently held positions (only after selling)
        i = assets.index(sym) if sym in assets else None
        p = float(probs[i]) if i is not None and np.isfinite(probs[i]) else float("nan")

        if min_hold_ok and np.isfinite(p) and p < thr_exit:
            actions_sell.append({"symbol": sym, "p": p})
            # record cooldown marker
            updated_pos[sym] = {
                "status": "cooldown",
                "cooldown_until": today_idx + int(args.cooldown_days),
                "last_action_idx": today_idx,
                "entry_idx": entry_idx,
            }
        else:
            actions_hold.append({"symbol": sym, "p": p})
            updated_pos[sym] = {
                "status": "open",
                "entry_idx": entry_idx,
                "last_action_idx": last_action_idx,
            }

    # Build BUY candidates
    open_syms = {s for s,info in updated_pos.items() if info.get("status") == "open"}
    cooldown_syms = {s for s,info in updated_pos.items() if info.get("status") == "cooldown" and today_idx < int(info.get("cooldown_until", -1))}
    capacity = max(0, int(args.max_positions) - len(open_syms))

    buy_cands = []
    if capacity > 0:
        for i, sym in enumerate(assets):
            if sym in open_syms or sym in cooldown_syms:
                continue
            p = float(probs[i])
            if not np.isfinite(p) or p < thr_enter:
                continue
            # compute vol on raw (unstandardized) close feature if present
            vol = float("nan")
            if "Close" in features:
                c_idx = features.index("Close")
                # use available closes in lookback window
                close_win = X[i, max(0, t_idx - 20):t_idx + 1, c_idx]
                vol = compute_recent_vol(close_win, lookback=20)
            score = rank_score(p, thr_enter, vol, mode=str(args.rank_mode))
            buy_cands.append({"symbol": sym, "p": p, "score": score, "vol": vol})

        buy_cands.sort(key=lambda d: d["score"], reverse=True)
        buy_cands = buy_cands[: min(capacity, int(args.max_new_per_day))]

    # Apply buys
    actions_buy = []
    for b in buy_cands:
        sym = b["symbol"]
        actions_buy.append(b)
        updated_pos[sym] = {"status": "open", "entry_idx": today_idx, "last_action_idx": today_idx}

    # Drop expired cooldown entries
    cleaned_pos: Dict[str, Dict] = {}
    for sym, info in updated_pos.items():
        if info.get("status") == "cooldown":
            if today_idx >= int(info.get("cooldown_until", -1)):
                continue
        cleaned_pos[sym] = info

    next_state = {"positions": cleaned_pos, "asof_date": asof_date, "asof_idx": today_idx}

    # Output report
    outdir = Path(args.outdir) / f"asof_{asof_date}"
    outdir.mkdir(parents=True, exist_ok=True)

    report = {
        "asof_date": asof_date,
        "asof_idx": today_idx,
        "n_assets": A,
        "thr_policy": args.thr_policy,
        "thr_enter": thr_enter,
        "thr_exit": thr_exit,
        "ensemble_mode": args.ensemble_mode,
        "n_models": len(models),
        "models": ckpt_paths,
        "buys": actions_buy,
        "sells": actions_sell,
        "holds": actions_hold[:50],  # cap for readability
        "positions_next": next_state,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(str(outdir / "report.json"), report)
    save_json(str(outdir / "positions_next.json"), next_state)

    if args.write_positions:
        save_json(args.positions, next_state)

    logger.info(f"wrote {outdir/'report.json'}")
    logger.info(f"buys={len(actions_buy)} sells={len(actions_sell)} open_next={sum(1 for v in cleaned_pos.values() if v.get('status')=='open')}")


if __name__ == "__main__":
    main()

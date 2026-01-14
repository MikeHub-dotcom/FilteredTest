#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/Test NN on (A,T,F) financial cube with explicit missingness mask.

Input files (from your generator):
- dataset_YYYY-MM-DD_masked.npy  : float32, NaNs already replaced by 0.0 placeholders
- mask_YYYY-MM-DD_masked.npy     : uint8 (1=observed, 0=missing)
- assets_YYYY-MM-DD.txt
- dates_YYYY-MM-DD.txt
- features_YYYY-MM-DD.json

Core idea:
- Model sees X and M: concat([X, M]) so missingness is explicit.
- Target y_t = 1 if forward return over H days minus estimated costs > 0 else 0
- Walk-forward time split to avoid leakage.
- Optimize decision threshold on validation to maximize win-rate / expected profit.

Dependencies:
  pip install numpy pandas torch
"""

import argparse
import json
import math
import os
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_log(msg: str) -> None:
    print(msg, flush=True)


def setup_logger(outdir: str) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(outdir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def masked_mean_std(X: np.ndarray, M: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean/std over (A,T) using mask M.
    X: (A,T,F) float32 (values, placeholder 0 in missing)
    M: (A,T,F) uint8 (1=observed,0=missing)
    Returns: mean(F,), std(F,)
    """
    M_f = M.astype(np.float32)
    denom = M_f.sum(axis=(0, 1)) + eps  # (F,)
    mean = (X * M_f).sum(axis=(0, 1)) / denom

    # variance
    diff = (X - mean[None, None, :]) * M_f
    var = (diff * diff).sum(axis=(0, 1)) / denom
    std = np.sqrt(var + eps)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_masked_standardize(X: np.ndarray, M: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    M_f = M.astype(np.float32)
    Xn = (X - mean[None, None, :]) / (std[None, None, :] + 1e-12)
    Xn = Xn * M_f  # missing -> 0
    return Xn.astype(np.float32, copy=False)


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    lookback: int = 128  # L
    horizon: int = 5  # H (forward return horizon)
    batch_size: int = 512
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    ignore_band_logret: float = 0.015  # 0.0 = deaktiviert

    # Cost model (very important for "profit probability")
    # roundtrip_cost_bps ~ commissions+spread+slippage (rough)
    roundtrip_cost_bps: float = 20.0  # 20 bps = 0.20% roundtrip

    # Validation threshold search
    min_trades_val: int = 100
    threshold_grid: int = 81  # number of thresholds between 0.5..0.9


# ----------------------------
# Dataset: pooled cross-asset samples
# ----------------------------
class CubeProfitDataset(Dataset):
    """
    Builds samples (X_seq, y, meta) from cube:
      X_seq: [C, L] float32 (C = 2F after concat X and M)
      y: 0/1 profit label (forward return after costs > 0)
    """

    def __init__(
            self,
            X: np.ndarray,  # (A,T,F) float32, NaNs already -> 0 placeholder
            M: np.ndarray,  # (A,T,F) uint8
            close_idx: int,  # feature index of "Close" in F
            t_start: int,
            t_end: int,  # inclusive end index for "t" anchor (we will use t as the end of lookback)
            lookback: int,
            horizon: int,
            roundtrip_cost_bps: float,
            close_raw: np.ndarray,  # (A,T)
            theta: float = 0.0,
            asset_filter: Optional[np.ndarray] = None,  # (A,) bool
            require_close_obs: bool = True,
    ):
        super().__init__()
        assert X.ndim == 3 and M.ndim == 3
        A, T, F = X.shape
        assert M.shape == (A, T, F)
        assert 0 <= close_idx < F

        self.X = X
        self.M = M
        self.A = A
        self.T = T
        self.F = F
        self.close_idx = close_idx
        self.close_raw = close_raw
        self.theta = float(theta)

        self.t_start = t_start
        self.t_end = t_end
        self.lookback = lookback
        self.horizon = horizon
        self.cost = roundtrip_cost_bps / 1e4  # bps -> fraction

        if asset_filter is None:
            asset_filter = np.ones((A,), dtype=bool)
        self.asset_filter = asset_filter.astype(bool)

        # Precompute valid (a,t) anchors to sample efficiently
        # Anchor t means input window [t-L+1..t] and target uses Close[t] -> Close[t+H]
        # Conditions:
        #  - t >= L-1
        #  - t+H < T
        #  - within [t_start..t_end]
        #  - optional: close observed at t and t+H (mask)
        anchors = []
        min_t = max(t_start, lookback - 1)
        max_t = min(t_end, T - horizon - 1)

        close_mask = self.M[:, :, close_idx]  # (A,T) uint8

        for a in range(A):
            if not self.asset_filter[a]:
                continue

            # iterate time anchors
            for t in range(min_t, max_t + 1):
                if require_close_obs:
                    if close_mask[a, t] == 0 or close_mask[a, t + horizon] == 0:
                        continue
                # Apply ignore band already in anchor construction (so we don't train on ignored samples) ---
                if self.theta > 0.0:
                    c0 = float(self.close_raw[a, t])
                    c1 = float(self.close_raw[a, t + horizon])
                    if c0 <= 0 or c1 <= 0:
                        continue
                    r_net = math.log(c1 / c0) - self.cost
                    if not (r_net > self.theta or r_net < -self.theta):
                        continue

                anchors.append((a, t))

        self.anchors = anchors

        # Small sanity check
        if len(self.anchors) == 0:
            raise RuntimeError("No valid samples after filtering/splitting. "
                               "Check lookback/horizon/split bounds/masks.")

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int):
        a, t = self.anchors[idx]
        L = self.lookback
        H = self.horizon

        # Input window
        x_win = self.X[a, t - L + 1: t + 1, :]  # (L,F)
        m_win = self.M[a, t - L + 1: t + 1, :].astype(np.float32)  # (L,F)

        # Concat features and mask along feature dimension => (L,2F)
        x_cat = np.concatenate([x_win, m_win], axis=1)  # (L,2F)

        # Profit label based on forward return after costs
        c0 = float(self.close_raw[a, t])
        c1 = float(self.close_raw[a, t + H])

        # If Close is missing, it would be 0 placeholder; mask should prevent selection.
        # Still guard against zeros:
        if c0 <= 0 or c1 <= 0:
            y = 0.0
            valid = 0.0
        else:
            r = math.log(c1 / c0)
            r_net = r - self.cost

            # After anchor filtering, all remaining samples are outside the ignore-band.
            y = 1.0 if (r_net > 0.0) else 0.0
            valid = 1.0

        # Return as tensors; CNN expects [C,L]
        x_t = torch.from_numpy(x_cat.T).float()  # (2F, L)
        y_t = torch.tensor([y], dtype=torch.float32)  # (1,)
        v_t = torch.tensor([valid], dtype=torch.float32)

        return x_t, y_t, v_t


# ----------------------------
# Model: simple 1D CNN
# ----------------------------
class ConvNet1D(nn.Module):
    def __init__(self, in_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),  # -> (B,64,1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),  # -> (B,64)
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        logit = self.head(z)
        return logit


# ----------------------------
# Metrics + Backtest
# ----------------------------
@torch.no_grad()
def evaluate_classifier(
        model: nn.Module,
        loader: DataLoader,
        device: str,
) -> Dict[str, float]:
    model.eval()
    logits_all = []
    y_all = []
    v_all = []

    for x, y, v in loader:
        x = x.to(device)
        y = y.to(device)
        v = v.to(device)
        logit = model(x)

        logits_all.append(logit.detach().cpu())
        y_all.append(y.detach().cpu())
        v_all.append(v.detach().cpu())

    logits = torch.cat(logits_all, dim=0).squeeze(1).numpy()
    y = torch.cat(y_all, dim=0).squeeze(1).numpy()
    v = torch.cat(v_all, dim=0).squeeze(1).numpy()

    # only valid rows
    mask = (v > 0.5)
    if mask.sum() == 0:
        return {"n": 0, "bce": float("nan"), "acc@0.5": float("nan")}

    logits = logits[mask]
    y = y[mask]

    p = 1.0 / (1.0 + np.exp(-logits))
    # BCE
    eps = 1e-9
    bce = float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean())
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    return {"n": int(mask.sum()), "bce": bce, "acc@0.5": acc}


@torch.no_grad()
def threshold_search_for_profitability(
    model: nn.Module,
    dataset: CubeProfitDataset,
    device: str,
    thresholds: np.ndarray,
    min_trades: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Choose threshold to maximize expected net log-return per trade on validation.
    Decision: trade if p > thr else flat.
    """
    model.eval()

    B = 2048
    n = len(dataset)

    probs = np.zeros((n,), dtype=np.float32)
    rnet = np.zeros((n,), dtype=np.float32)
    v = np.zeros((n,), dtype=np.float32)

    for i in range(0, n, B):
        batch = [dataset[j] for j in range(i, min(i + B, n))]
        x = torch.stack([b[0] for b in batch], dim=0).to(device)

        # compute p
        logit = model(x).detach().cpu().numpy().squeeze(1)
        p = 1.0 / (1.0 + np.exp(-logit))
        probs[i:i + len(batch)] = p.astype(np.float32)

        v[i:i + len(batch)] = 1.0

        for k, (a, t) in enumerate(dataset.anchors[i:i + len(batch)]):
            c0 = float(dataset.close_raw[a, t])
            c1 = float(dataset.close_raw[a, t + dataset.horizon])
            if c0 <= 0.0 or c1 <= 0.0:
                rnet[i + k] = 0.0
                v[i + k] = 0.0
            else:
                rnet[i + k] = float(math.log(c1 / c0) - dataset.cost)

    valid = (v > 0.5)
    probs = probs[valid]
    rnet = rnet[valid]

    best_thr = 0.5
    best_score = -1e9
    best_info: Dict[str, float] = {}

    for thr in thresholds:
        take = probs > thr
        trades = int(take.sum())
        if trades < min_trades:
            continue

        avg_r = float(rnet[take].mean())
        winrate = float((rnet[take] > 0.0).mean())

        # score: expected net return per trade (primary), small penalty for very low trade count
        score = avg_r - 0.0001 * (1.0 / math.sqrt(trades))

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_info = {
                "trades": trades,
                "avg_rnet": avg_r,
                "winrate": winrate,
                "score": float(score),
            }

    if not best_info:
        take = probs > 0.5
        trades = int(take.sum())
        avg_r = float(rnet[take].mean()) if trades > 0 else float("nan")
        winrate = float((rnet[take] > 0.0).mean()) if trades > 0 else float("nan")
        best_thr = 0.5
        best_info = {"trades": trades, "avg_rnet": avg_r, "winrate": winrate, "score": float("nan")}

    return best_thr, best_info


# ----------------------------
# Train loop
# ----------------------------
def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        opt: torch.optim.Optimizer,
        device: str,
        logger: logging.Logger,
        epoch: int,
        log_every: int = 100,  # <<< statt 1
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    losses = []
    ema = None
    alpha = 0.05  # EMA smoothing
    t0 = time.time()

    for step, (x, y, v) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logit = model(x)
        logit = torch.clamp(logit, -10.0, 10.0)  # stabilizes BCE under noisy labels

        loss_raw = loss_fn(logit, y)
        loss = (loss_raw * v).sum() / (v.sum() + 1e-9)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)

        # EMA for stable logging
        ema = loss_val if ema is None else (1 - alpha) * ema + alpha * loss_val

        if (step % log_every) == 0 or step == 1:
            lr = opt.param_groups[0]["lr"]
            elapsed = time.time() - t0
            step_s = step / max(elapsed, 1e-9)
            # label balance in this batch (valid only)
            with torch.no_grad():
                vb = (v > 0.5).squeeze(1)
                if vb.any():
                    y_mean = float(y[vb].mean().detach().cpu().item())
                else:
                    y_mean = float("nan")
            logger.info(
                f"E{epoch:03d} S{step:05d}/{len(loader):05d} "
                f"loss={loss_val:.6f} ema={ema:.6f} y_mean={y_mean:.3f} "
                f"lr={lr:.3e} step/s={step_s:.2f}"
            )

    return float(np.mean(losses)) if losses else float("nan")


# ----------------------------
# Split logic
# ----------------------------
def time_split_indices(T: int, train_frac: float = 0.70, val_frac: float = 0.15) -> Tuple[
    Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Returns (train_start, train_end), (val_start, val_end), (test_start, test_end)
    indices are inclusive.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    t_train_end = int(T * train_frac) - 1
    t_val_end = t_train_end + int(T * val_frac)
    t_val_end = min(t_val_end, T - 1)
    return (0, t_train_end), (t_train_end + 1, t_val_end), (t_val_end + 1, T - 1)


def build_asset_filter_from_masks(
        M: np.ndarray,
        close_idx: int,
        min_obs: int = 750,
        max_missing_close: float = 0.05,
        max_gap: int = 5,
) -> np.ndarray:
    """
    Simple, robust filter using mask only (no price values needed).
    Operates on the asset's existence window: first..last observed Close.
    """
    A, T, F = M.shape
    close_m = M[:, :, close_idx].astype(bool)

    keep = np.zeros((A,), dtype=bool)

    for a in range(A):
        idx = np.where(close_m[a])[0]
        if idx.size == 0:
            continue
        first, last = int(idx[0]), int(idx[-1])
        w = close_m[a, first:last + 1]  # True=observed

        obs = int(w.sum())
        if obs < min_obs:
            continue

        missing_rate = float((~w).mean())
        if missing_rate > max_missing_close:
            continue

        # max consecutive missing
        is_missing = (~w).astype(np.uint8)
        max_run, run = 0, 0
        for m in is_missing:
            run = run + 1 if m else 0
            max_run = max(max_run, run)
        if max_run > max_gap:
            continue

        keep[a] = True

    return keep


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset_*_masked.npy")
    ap.add_argument("--mask", required=True, help="Path to mask_*_masked.npy")
    ap.add_argument("--assets", required=True, help="Path to assets_*.txt")
    ap.add_argument("--dates", required=True, help="Path to dates_*.txt")
    ap.add_argument("--features", required=True, help="Path to features_*.json")
    ap.add_argument("--outdir", default="./runs", help="Output directory")
    ap.add_argument("--asset_name", default="",
                    help="Optional: evaluate/focus on single asset name (must exist in assets list)")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(args.outdir, exist_ok=True)

    logger = setup_logger(args.outdir)
    logger.info(f"device={cfg.device}")

    X = np.load(args.dataset)  # (A,T,F) float32
    M = np.load(args.mask)  # (A,T,F) uint8
    assets = load_text_lines(args.assets)
    dates = load_text_lines(args.dates)
    feat_names = load_json(args.features)

    assert X.shape == M.shape
    A, T, F = X.shape
    safe_log(f"[INFO] loaded cube: A={A}, T={T}, F={F}")

    # Identify Close column
    close_idx = None
    for i, n in enumerate(feat_names):
        if str(n).strip().lower() == "close":
            close_idx = i
            break
    if close_idx is None:
        raise RuntimeError("Could not find 'Close' feature in features json.")

    # Time splits (inclusive indices for anchor t)
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = time_split_indices(T, args.train_frac, args.val_frac)
    safe_log(f"[INFO] time split: train=[{tr_s},{tr_e}] val=[{va_s},{va_e}] test=[{te_s},{te_e}]")
    safe_log(
        f"[INFO] date ranges: train {dates[tr_s]}..{dates[tr_e]} | val {dates[va_s]}..{dates[va_e]} | test {dates[te_s]}..{dates[te_e]}")

    close_raw = X[:, :, close_idx].copy()  # (A,T) Rohpreise für Labels

    # --- NEW: compute normalization stats on TRAIN WINDOW ONLY (leakage-free) ---
    X_train = X[:, tr_s:tr_e + 1, :]
    M_train = M[:, tr_s:tr_e + 1, :]

    mu, sig = masked_mean_std(X_train, M_train)

    # apply to full X (train/val/test) consistently
    X = apply_masked_standardize(X, M, mu, sig)
    safe_log("[INFO] applied masked standardization using train-only statistics")

    # Asset filter (optional but recommended even though you already gated in build)
    asset_keep = build_asset_filter_from_masks(M, close_idx=close_idx)
    safe_log(f"[INFO] asset_keep: {asset_keep.sum()}/{A} assets remain after mask-only filter")

    # Optional: focus on a single asset for evaluation (still train cross-asset unless you want to change it)
    focus_asset_idx = None
    if args.asset_name:
        if args.asset_name not in assets:
            raise RuntimeError(f"asset_name '{args.asset_name}' not found in assets list.")
        focus_asset_idx = assets.index(args.asset_name)
        safe_log(f"[INFO] focus asset: {args.asset_name} (idx={focus_asset_idx})")

    # Build datasets
    train_ds = CubeProfitDataset(
        X=X, M=M, close_idx=close_idx,
        t_start=tr_s, t_end=tr_e,
        lookback=cfg.lookback, horizon=cfg.horizon,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        asset_filter=asset_keep,
        require_close_obs=True,
        close_raw=close_raw,
        theta=cfg.ignore_band_logret
    )
    val_ds = CubeProfitDataset(
        X=X, M=M, close_idx=close_idx,
        t_start=va_s, t_end=va_e,
        lookback=cfg.lookback, horizon=cfg.horizon,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        asset_filter=asset_keep,
        require_close_obs=True,
        close_raw=close_raw,
        theta=cfg.ignore_band_logret
    )
    test_ds = CubeProfitDataset(
        X=X, M=M, close_idx=close_idx,
        t_start=te_s, t_end=te_e,
        lookback=cfg.lookback, horizon=cfg.horizon,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        asset_filter=asset_keep,
        require_close_obs=True,
        close_raw=close_raw,
        theta=cfg.ignore_band_logret
    )

    safe_log(f"[INFO] samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pin = (cfg.device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=pin)

    # Model
    in_ch = 2 * F
    model = ConvNet1D(in_ch=in_ch, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val_avg_rnet = -float("inf")
    best_path = os.path.join(args.outdir, "best_model.pt")

    logger.info("training start")
    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()

        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            opt=opt,
            device=cfg.device,
            logger=logger,
            epoch=epoch,
            log_every=100,
        )

        val_metrics = evaluate_classifier(model, val_loader, cfg.device)

        # --- NEW: profit-oriented validation metrics (do NOT rely on BCE/acc for trading) ---
        thresholds = np.linspace(0.50, 0.90, cfg.threshold_grid, dtype=np.float32)
        val_thr, val_info = threshold_search_for_profitability(
            model=model,
            dataset=val_ds,
            device=cfg.device,
            thresholds=thresholds,
            min_trades=cfg.min_trades_val,
        )

        val_trades = int(val_info.get("trades", 0))
        val_avg_rnet = float(val_info.get("avg_rnet", float("nan")))
        val_winrate = float(val_info.get("winrate", float("nan")))

        logger.info(
            f"[VAL] thr={val_thr:.3f} trades={val_trades} "
            f"avg_rnet={val_avg_rnet:.6f} winrate={val_winrate:.3f}"
        )

        ep_time = time.time() - ep_t0
        logger.info(
            f"E{epoch:03d} DONE "
            f"train_loss={tr_loss:.6f} "
            f"val_bce={val_metrics['bce']:.6f} val_acc@0.5={val_metrics['acc@0.5']:.4f} "
            f"n_val={val_metrics['n']} "
            f"epoch_time_s={ep_time:.1f}"
        )

        # --- NEW: checkpoint by profitability proxy, not BCE ---
        if np.isfinite(val_avg_rnet) and val_trades >= cfg.min_trades_val and val_avg_rnet > best_val_avg_rnet:
            best_val_avg_rnet = val_avg_rnet
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "val_thr": val_thr,
                    "val_info": val_info,
                    "cfg": asdict(cfg),
                },
                best_path,
            )
            logger.info(
                f"[CKPT] saved {os.path.basename(best_path)} "
                f"(best_val_avg_rnet={best_val_avg_rnet:.6f}, trades={val_trades}, thr={val_thr:.3f})"
            )

        sched.step()

    safe_log(f"[INFO] best model saved: {best_path} (best_val_avg_rnet={best_val_avg_rnet:.6f})")

    # Reload best
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])

    # Threshold optimization to maximize profit probability among taken trades
    thresholds = np.linspace(0.50, 0.90, cfg.threshold_grid, dtype=np.float32)
    thr, info = threshold_search_for_profitability(
        model=model,
        dataset=val_ds,
        device=cfg.device,
        thresholds=thresholds,
        min_trades=cfg.min_trades_val,
    )
    logger.info(
        f"[VAL] chosen threshold: thr={thr:.3f} "
        f"| trades={info.get('trades')} "
        f"| winrate={info.get('winrate'):.3f} "
        f"| avg_rnet={info.get('avg_rnet'):.6f}"
    )

    # Report test metrics (classifier)
    test_metrics = evaluate_classifier(model, test_loader, cfg.device)
    safe_log(f"[TEST] bce={test_metrics['bce']:.4f} acc@0.5={test_metrics['acc@0.5']:.4f} n_test={test_metrics['n']}")

    # Simple backtest proxy on test: winrate among taken trades
    thr_test, info_test = threshold_search_for_profitability(
        model=model,
        dataset=test_ds,
        device=cfg.device,
        thresholds=np.array([thr], dtype=np.float32),
        min_trades=0,
    )
    safe_log(
        f"[TEST] trading@thr={thr:.3f} | trades={info_test.get('trades')} | winrate={info_test.get('winrate'):.3f}")

    safe_log("[DONE]")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # IDE RUN CONFIG (set your paths here)
    # ------------------------------------------------------------
    ROOT_DIR = str(Path(__file__).resolve().parent)

    DATASET_NPY = ROOT_DIR + "/dataset/dataset_2026-01-13_masked.npy"
    MASK_NPY = ROOT_DIR + "/dataset/mask_2026-01-13_masked.npy"
    ASSETS_TXT = ROOT_DIR + "/dataset/assets_2026-01-13.txt"
    DATES_TXT = ROOT_DIR + "/dataset/dates_2026-01-13.txt"
    FEATURES_JS = ROOT_DIR + "/dataset/features_2026-01-13.json"

    OUTDIR = ROOT_DIR + "/runs"
    FOCUS_ASSET = ""  # e.g. "DRO.AX" or "" to disable

    # Optional: override split fractions (must satisfy train+val < 1)
    TRAIN_FRAC = 0.70
    VAL_FRAC = 0.15

    # Optional: quick sanity checks
    for p in [DATASET_NPY, MASK_NPY, ASSETS_TXT, DATES_TXT, FEATURES_JS]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    # ------------------------------------------------------------
    # Emulate argparse.Namespace and call main() logic
    # ------------------------------------------------------------
    import sys

    sys.argv = [
        sys.argv[0],
        "--dataset", DATASET_NPY,
        "--mask", MASK_NPY,
        "--assets", ASSETS_TXT,
        "--dates", DATES_TXT,
        "--features", FEATURES_JS,
        "--outdir", OUTDIR,
        "--train_frac", str(TRAIN_FRAC),
        "--val_frac", str(VAL_FRAC),
    ]

    if FOCUS_ASSET:
        sys.argv += ["--asset_name", FOCUS_ASSET]

    main()

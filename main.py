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
import copy
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless saving (no GUI)
import matplotlib.pyplot as plt

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- AMP compatibility (PyTorch >= 2.0 prefers torch.amp.*) ---
try:
    from torch.amp import autocast as amp_autocast
    from torch.amp import GradScaler as AmpGradScaler
except Exception:  # fallback for older torch
    from torch.cuda.amp import autocast as amp_autocast
    from torch.cuda.amp import GradScaler as AmpGradScaler


def amp_device_type(device: str) -> str:
    # torch.amp wants "cuda"/"cpu" etc.
    return "cuda" if str(device).startswith("cuda") else "cpu"



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
    # Early stopping (based on validation trading metric, not BCE)
    early_stop_patience: int = 12  # epochs without improvement until stop; 0 disables
    early_stop_min_epochs: int = 30  # don't stop before this epoch
    ignore_band_logret: float = 0.015  # 0.0 = deaktiviert

    # Cost model (very important for "profit probability")
    # roundtrip_cost_bps ~ commissions+spread+slippage (rough)
    roundtrip_cost_bps: float = 20.0  # 20 bps = 0.20% roundtrip

    # Validation threshold search
    # Validation threshold search
    min_trades_val: int = 100
    threshold_grid: int = 81  # number of thresholds between thr_min..thr_max
    thr_min: float = 0.50
    thr_max: float = 0.99

    # Fast sweep / screening
    max_anchors_train: int = 1_500_000  # set 0 to disable
    max_anchors_val: int = 600_000  # set 0 to disable
    seed_trials: int = 123

    # GPU speed-ups
    amp: bool = True
    cudnn_benchmark: bool = True
    grad_clip: float = 1.0

    # --- Model selection ---
    model_type: str = "tcn"   # "cnn" (old ConvNet1D) or "tcn"

    # --- TCN hyperparameters (finance-robust defaults) ---
    tcn_channels: int = 192        # base width
    tcn_levels: int = 5            # number of residual blocks (dilations 1,2,4,...)
    tcn_kernel_size: int = 5       # 3/5/7
    tcn_dropout: float = 0.10      # separate dropout for TCN (often works better than sharing cfg.dropout)
    tcn_use_groupnorm: bool = True # GN tends to be stable with large batch / non-stationary signals

    # --- PatchTST (patched Transformer for time series) ---
    # Patch along time axis -> Transformer encoder -> CLS head
    patch_len: int = 32            # patch length along time axis
    patch_stride: int = 16         # stride between patches (<= patch_len)
    d_model: int = 256             # token embedding dimension
    nhead: int = 8                 # attention heads
    num_layers: int = 4            # Transformer encoder layers
    dim_feedforward: int = 1024    # FFN hidden dim
    norm_first: bool = True        # Pre-LN tends to be stabler

    # --- iTransformer (Inverted Transformer: Tokens=Variablen) ---
    itr_use_cls: bool = True  # CLS-Token statt Pooling
    itr_pool: str = "mean"  # falls itr_use_cls=False: "mean" | "max"



def sample_trial_cfg_tcn(base: Config, rng: np.random.Generator) -> Config:
    """
    Architektur- und signal-fokussierter Sweep für Finanzdaten.
    Ziel: robuste Generalisierung + sinnvolle Trade-Frequenz.
    """
    cfg = Config(**asdict(base))

    # ----------------------------
    # 1) Model choice (prioritize TCN)
    # ----------------------------
    cfg.model_type = str(rng.choice(["tcn"]))  # TCN only in dieser Konfiguration

    # ----------------------------
    # 2) Signal / trading-relevant params
    # ----------------------------
    cfg.horizon = int(rng.choice([3, 5, 7, 10, 15, 20]))
    cfg.lookback = int(rng.choice([
        96, 128, 192, 256,
        384, 384,
        512, 512,
        640,
        768, 768,
        1024
    ]))
    cfg.ignore_band_logret = float(rng.choice([0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030]))
    cfg.roundtrip_cost_bps = float(rng.choice([5.0, 10.0, 15.0, 20.0, 35.0, 50.0, 75.0, 100.0]))

    # ----------------------------
    # 3) Optimization
    # ----------------------------
    # TCN tolerates a bit higher LR than transformer-like models; keep conservative set
    cfg.lr = float(rng.choice([2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3, 1.6e-3]))
    cfg.weight_decay = float(rng.choice([0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]))
    cfg.grad_clip = float(rng.choice([0.5, 1.0, 2.0, 5.0]))

    # Batch (4090 friendly); keep not too large to avoid dataloader overhead
    cfg.batch_size = int(rng.choice([256, 512, 768, 1024]))

    # ----------------------------
    # 4) Model regularization
    # ----------------------------
    cfg.dropout = float(rng.choice([0.00, 0.05, 0.10, 0.20, 0.30]))  # used by CNN head
    cfg.tcn_dropout = float(rng.choice([0.00, 0.05, 0.10, 0.15, 0.20, 0.30]))

    # ----------------------------
    # 5) TCN architecture params (if model_type=tcn)
    # ----------------------------
    # Receptive field grows with levels and dilation; these ranges are good for daily OHLCV
    cfg.tcn_channels = int(rng.choice([128, 192, 256, 384, 512]))
    cfg.tcn_levels = int(rng.choice([4, 5, 6, 7, 8, 9]))
    cfg.tcn_kernel_size = int(rng.choice([3, 5, 7, 9]))
    cfg.tcn_use_groupnorm = True

    # ----------------------------
    # 6) Validation threshold search params (avoid "lucky few trades")
    # ----------------------------
    cfg.threshold_grid = 81
    cfg.min_trades_val = 200

    # ----------------------------
    # 7) Sweep speed (screening)
    # ----------------------------
    cfg.epochs = 2  # overwritten by args.sweep_epochs at runtime
    cfg.max_anchors_train = int(rng.choice([800_000, 1_200_000, 1_500_000, 2_000_000]))
    cfg.max_anchors_val = int(rng.choice([300_000, 450_000, 600_000, 900_000]))

    # OOM Guardrail
    # Guard rails: avoid pathological (lookback, batch) combos causing OOM
    if cfg.lookback >= 512:
        cfg.batch_size = int(min(cfg.batch_size, 256))
    if cfg.lookback >= 768:
        cfg.batch_size = int(min(cfg.batch_size, 128))

    return cfg


def sample_trial_cfg(base_cfg: Config, rng: np.random.Generator, sweep_arch: str = "mix") -> Config:
    cfg = copy.deepcopy(base_cfg)

    # ---- shared / training ----
    cfg.lr = float(10 ** rng.uniform(-4.6, -3.0))                 # ~2.5e-5 .. 1e-3
    cfg.weight_decay = float(10 ** rng.uniform(-8.0, -3.8))       # very small .. moderate
    cfg.dropout = float(rng.uniform(0.00, 0.25))
    cfg.label_smooth = float(rng.uniform(0.00, 0.10))

    # Open lookback (previous sweep dominated by max -> open upward)
    cfg.lookback = int(rng.choice([256, 384, 512, 768, 1024, 1536, 2048]))

    # anchors (keep, but avoid extreme tiny values)
    cfg.max_anchors_train = int(rng.choice([4000, 8000, 12000, 20000, 0]))
    cfg.max_anchors_val   = int(rng.choice([4000, 8000, 12000, 0]))

    # ---- choose architecture family ----
    if sweep_arch == "patchtst":
        cfg.model_type = "patchtst"
    elif sweep_arch == "itransformer":
        cfg.model_type = "itransformer"
    else:
        cfg.model_type = str(rng.choice(["patchtst", "itransformer"]))

    # ---- PatchTST search space ----
    if cfg.model_type == "patchtst":
        cfg.patch_len = int(rng.choice([8, 16, 32, 64]))
        cfg.patch_stride = int(rng.choice([cfg.patch_len // 2, cfg.patch_len]))  # 50% overlap or none

        cfg.tr_d_model = int(rng.choice([64, 96, 128, 192, 256]))
        cfg.tr_n_heads = int(rng.choice([4, 8]))
        cfg.tr_n_layers = int(rng.choice([2, 3, 4, 5, 6]))
        cfg.tr_ff_mult = int(rng.choice([2, 4, 6]))
        cfg.tr_dropout = float(rng.uniform(0.00, 0.25))
        cfg.tr_attn_dropout = float(rng.uniform(0.00, 0.20))

    # ---- iTransformer search space ----
    else:  # itransformer
        cfg.tr_d_model = int(rng.choice([64, 96, 128, 192, 256]))
        cfg.tr_n_heads = int(rng.choice([4, 8]))
        cfg.tr_n_layers = int(rng.choice([2, 3, 4, 6, 8]))
        cfg.tr_ff_mult = int(rng.choice([2, 4, 6]))
        cfg.tr_dropout = float(rng.uniform(0.00, 0.25))
        cfg.tr_attn_dropout = float(rng.uniform(0.00, 0.20))

        # no patching here (inverted tokens); keep patch params default/unused

    return cfg



def make_trial_outdir(base_outdir: str, trial_id: int) -> str:
    d = os.path.join(base_outdir, f"trial_{trial_id:04d}")
    os.makedirs(d, exist_ok=True)
    return d


def write_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_existing_sweep_results(results_path: str):
    """
    Parse existing sweep results.jsonl and return:
      completed: set of (trial:int, stage:str, rep:int) that already exist
      by_key: dict[(trial,stage,rep)] -> row (full dict), used to reuse metrics/time
    Safe for partial/corrupt last line.
    """
    completed = set()
    by_key = {}
    if not results_path or not os.path.exists(results_path):
        return completed, by_key

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                # tolerate partial last line after crash/reboot
                continue

            if "trial" not in row or "stage" not in row or "rep" not in row:
                continue

            try:
                k = (int(row["trial"]), str(row["stage"]), int(row["rep"]))
            except Exception:
                continue

            completed.add(k)
            by_key[k] = row

    return completed, by_key


def run_single_training(
    cfg: Config,
    args,
    X: np.ndarray,
    M: np.ndarray,
    assets: List[str],
    dates: List[str],
    feat_names: List[str],
    close_idx: int,
    splits: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    outdir: str,
) -> Dict[str, float]:
    """
    Runs train/val for cfg.epochs and returns val profit metrics + classifier metrics.
    Uses subsampling via cfg.max_anchors_* for sweep speed.
    """
    logger = setup_logger(outdir)
    logger.info(f"device={cfg.device}")
    # IMPORTANT: use cfg.seed for model init / dataloader randomness
    # cfg.seed_trials is reserved for dataset anchor subsampling reproducibility
    set_seed(int(cfg.seed))
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    scaler = AmpGradScaler(amp_device_type(cfg.device), enabled=(cfg.amp and str(cfg.device).startswith("cuda")))

    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = splits
    A, T, F = X.shape

    close_raw = X[:, :, close_idx].copy()

    # leakage-free standardization (train window only)
    X_train = X[:, tr_s:tr_e + 1, :]
    M_train = M[:, tr_s:tr_e + 1, :]
    mu, sig = masked_mean_std(X_train, M_train)
    Xn = apply_masked_standardize(X, M, mu, sig)

    asset_keep = build_asset_filter_from_masks(M, close_idx=close_idx)

    train_ds = CubeProfitDataset(
        X=Xn, M=M, close_idx=close_idx,
        t_start=tr_s, t_end=tr_e,
        lookback=cfg.lookback, horizon=cfg.horizon,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        asset_filter=asset_keep,
        require_close_obs=True,
        close_raw=close_raw,
        theta=cfg.ignore_band_logret,
        max_anchors=cfg.max_anchors_train,
        seed=cfg.seed_trials,
    )
    val_ds = CubeProfitDataset(
        X=Xn, M=M, close_idx=close_idx,
        t_start=va_s, t_end=va_e,
        lookback=cfg.lookback, horizon=cfg.horizon,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        asset_filter=asset_keep,
        require_close_obs=True,
        close_raw=close_raw,
        theta=cfg.ignore_band_logret,
        max_anchors=cfg.max_anchors_val,
        seed=cfg.seed_trials + 1,
    )

    pin = (cfg.device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)

    model = build_model(cfg, in_ch=2 * F).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val_avg_rnet = -float("inf")
    best_thr = 0.5
    best_info = {"trades": 0, "avg_rnet": float("nan"), "winrate": float("nan")}
    best_val_bce = float("nan")
    best_val_acc = float("nan")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(
            model=model, loader=train_loader, opt=opt, device=cfg.device,
            logger=logger, epoch=epoch, scaler=scaler, amp=cfg.amp,
            grad_clip=cfg.grad_clip, log_every=200
        )
        val_metrics = evaluate_classifier(model, val_loader, cfg.device)

        thresholds = np.linspace(cfg.thr_min, cfg.thr_max, cfg.threshold_grid, dtype=np.float32)
        val_thr, val_info = threshold_search_for_profitability(
            model=model, dataset=val_ds, device=cfg.device,
            thresholds=thresholds, min_trades=cfg.min_trades_val
        )

        val_trades = int(val_info.get("trades", 0))
        val_avg_rnet = float(val_info.get("avg_rnet", float("nan")))

        if np.isfinite(val_avg_rnet) and val_trades >= cfg.min_trades_val and val_avg_rnet > best_val_avg_rnet:
            best_val_avg_rnet = val_avg_rnet
            best_thr = float(val_thr)
            best_info = dict(val_info)
            best_val_bce = float(val_metrics["bce"])
            best_val_acc = float(val_metrics["acc@0.5"])

        sched.step()

    return {
        "best_val_avg_rnet": float(best_val_avg_rnet),
        "best_val_thr": float(best_thr),
        "best_val_trades": int(best_info.get("trades", 0)),
        "best_val_winrate": float(best_info.get("winrate", float("nan"))),
        "best_val_score": float(best_info.get("score", float("nan"))),
        "best_val_bce": float(best_val_bce),
        "best_val_acc": float(best_val_acc),
    }


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
            max_anchors: int = 0,
            seed: int = 0,
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
        self.max_anchors = int(max_anchors)
        self.seed = int(seed)

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

                # --- NEW: ignore-band filtering at anchor-build time (fast) ---
                if self.theta > 0.0:
                    c0 = float(self.close_raw[a, t])
                    c1 = float(self.close_raw[a, t + horizon])
                    if c0 <= 0.0 or c1 <= 0.0:
                        continue
                    r_net = math.log(c1 / c0) - self.cost
                    if not (r_net > self.theta or r_net < -self.theta):
                        continue

                anchors.append((a, t))

        self.anchors = anchors

        # --- NEW: optional anchor subsampling for fast sweeps ---
        if self.max_anchors > 0 and len(self.anchors) > self.max_anchors:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(self.anchors), size=self.max_anchors, replace=False)
            self.anchors = [self.anchors[i] for i in idx]

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


class ResidualTCNBlock(nn.Module):
    """
    Finance-robuster TCN Block:
    - two dilated convs
    - residual connection (with 1x1 projection if needed)
    - GroupNorm (stable) + GELU
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
        padding = (kernel_size - 1) * dilation // 2  # keep length
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)

        if use_groupnorm:
            # groups: small number for stability; ensure divisibility
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


class TCNModel(nn.Module):
    """
    TCN for (B, C, L) with explicit mask-channel included in C.
    - dilations grow exponentially -> long receptive field
    - adaptive pooling -> single logit
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
        blocks = []
        ch_in = in_ch
        for i in range(levels):
            d = 2 ** i
            blocks.append(
                ResidualTCNBlock(
                    in_ch=ch_in,
                    out_ch=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    use_groupnorm=use_groupnorm,
                )
            )
            ch_in = channels

        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B,channels,1)

        # small head; keep it simple for finance robustness
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.pool(z)
        return self.head(z)


class PatchTSTModel(nn.Module):
    """
    PatchTST-style Transformer encoder over time patches.

    Input: x of shape (B, C, L) with C=features (+mask channel if enabled).
    Output: logits of shape (B, 1)
    """

    def __init__(
        self,
        in_ch: int,
        lookback: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool = True,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.lookback = int(lookback)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)

        if self.patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be > 0")
        if self.patch_stride > self.patch_len:
            raise ValueError("patch_stride must be <= patch_len")

        # number of patches for the configured lookback (pad if needed)
        if self.lookback < self.patch_len:
            self.n_patches = 1
        else:
            self.n_patches = 1 + (self.lookback - self.patch_len) // self.patch_stride

        self.token_dim = self.in_ch * self.patch_len
        self.proj = nn.Linear(self.token_dim, d_model)

        # CLS token + positional embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _make_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) -> patches: (B, N, C*patch_len)
        Pads on the right so that we can extract the last patch cleanly.
        """
        B, C, L = x.shape
        if L < self.patch_len:
            pad = self.patch_len - L
            x = F.pad(x, (0, pad))
            L = x.shape[-1]

        # ensure enough length to cover n_patches with stride
        last_start = (self.n_patches - 1) * self.patch_stride
        need_L = last_start + self.patch_len
        if L < need_L:
            x = F.pad(x, (0, need_L - L))
            L = x.shape[-1]

        patches = []
        for i in range(self.n_patches):
            s = i * self.patch_stride
            e = s + self.patch_len
            p = x[:, :, s:e]                 # (B, C, patch_len)
            patches.append(p.reshape(B, -1)) # (B, C*patch_len)
        return torch.stack(patches, dim=1)   # (B, N, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        patches = self._make_patches(x)              # (B, N, token_dim)
        z = self.proj(patches)                       # (B, N, d_model)

        cls = self.cls.expand(z.size(0), -1, -1)     # (B, 1, d_model)
        z = torch.cat([cls, z], dim=1)               # (B, 1+N, d_model)
        z = z + self.pos[:, : z.shape[1], :]
        z = self.dropout(z)
        z = self.encoder(z)
        z = self.norm(z)
        cls_out = z[:, 0, :]
        return self.head(cls_out)


class ITransformerModel(nn.Module):
    """
    iTransformer (inverted): Tokens sind Variablen/Features (C Tokens),
    jede Variable wird durch ihre Lookback-Historie (L) via Linear(L->d_model)
    eingebettet. Danach Transformer-Encoder über Variablen.
    Output: Logit pro Sample (Binary-Decision).
    """
    def __init__(self, in_ch: int, lookback: int, cfg: "Config"):
        super().__init__()
        self.in_ch = int(in_ch)
        self.lookback = int(lookback)
        self.d_model = int(cfg.d_model)

        # Projektion der Zeitachse pro Variable: (B,C,L) -> (B,C,d_model)
        self.var_proj = nn.Linear(self.lookback, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(cfg.nhead),
            dim_feedforward=int(cfg.dim_feedforward),
            dropout=float(cfg.dropout),
            batch_first=True,
            norm_first=bool(cfg.norm_first),
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.num_layers))
        self.norm = nn.LayerNorm(self.d_model)

        self.use_cls = bool(getattr(cfg, "itr_use_cls", True))
        self.pool = str(getattr(cfg, "itr_pool", "mean")).lower()

        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
            tok_count = self.in_ch + 1
        else:
            self.cls = None
            tok_count = self.in_ch

        # Positions/Token-Embedding über Variablen (nicht über Zeit)
        self.pos = nn.Parameter(torch.zeros(1, tok_count, self.d_model))
        self.dropout = nn.Dropout(float(cfg.dropout))

        # Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(self.d_model, 1),
        )

        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        if self.cls is not None:
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        if x.dim() != 3:
            raise ValueError(f"expected x shape (B,C,L), got {tuple(x.shape)}")

        B, C, L = x.shape
        if C != self.in_ch:
            # robust, aber eigentlich sollte C konstant sein
            if C > self.in_ch:
                x = x[:, : self.in_ch, :]
                C = self.in_ch
            else:
                pad_c = self.in_ch - C
                x = torch.cat([x, x.new_zeros((B, pad_c, L))], dim=1)
                C = self.in_ch

        # Lookback auf exakt self.lookback bringen (truncate left / pad left)
        if L != self.lookback:
            if L > self.lookback:
                x = x[:, :, -self.lookback :]
            else:
                x = F.pad(x, (self.lookback - L, 0))

        # Variable-Tokens erzeugen
        z = self.var_proj(x)  # (B, C, d_model)

        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)  # (B,1,d_model)
            z = torch.cat([cls, z], dim=1)   # (B,1+C,d_model)

        z = z + self.pos[:, : z.size(1), :]
        z = self.dropout(z)

        z = self.encoder(z)
        z = self.norm(z)

        if self.use_cls:
            h = z[:, 0, :]
        else:
            if self.pool == "max":
                h = z.max(dim=1).values
            else:
                h = z.mean(dim=1)

        return self.head(h)



def build_model(cfg: Config, in_ch: int) -> nn.Module:
    mt = str(cfg.model_type).lower().strip()
    if mt == "cnn":
        return ConvNet1D(in_ch=in_ch, dropout=cfg.dropout)
    if mt == "tcn":
        return TCNModel(
            in_ch=in_ch,
            channels=int(cfg.tcn_channels),
            levels=int(cfg.tcn_levels),
            kernel_size=int(cfg.tcn_kernel_size),
            dropout=float(cfg.tcn_dropout),
            use_groupnorm=bool(cfg.tcn_use_groupnorm),
        )
    if mt == "patchtst":
        return PatchTSTModel(
            in_ch=in_ch,
            lookback=int(cfg.lookback),
            patch_len=int(cfg.patch_len),
            patch_stride=int(cfg.patch_stride),
            d_model=int(cfg.d_model),
            nhead=int(cfg.nhead),
            num_layers=int(cfg.num_layers),
            dim_feedforward=int(cfg.dim_feedforward),
            dropout=float(cfg.dropout),
            norm_first=bool(cfg.norm_first),
        )
    if mt in ("itransformer", "itr", "i_transformer"):
        return ITransformerModel(
            in_ch=in_ch,
            lookback=int(cfg.lookback),
            cfg=cfg,
        )
    raise ValueError(f"Unknown model_type='{cfg.model_type}'. Use 'cnn' or 'tcn', 'patchtst' or 'itransformer'.")


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
# TEST REPORT + REALISTIC TRADING SIM
# ----------------------------
def _project_root_from_dataset_path(dataset_path: str) -> str:
    # dataset/dataset_*.npy -> project root
    p = Path(dataset_path).resolve()
    # if dataset folder exists in parent, treat parent as root
    if p.parent.name.lower() == "dataset":
        return str(p.parent.parent)
    return str(p.parent)


def load_isin_name_map(misc_dir: str) -> Dict[str, str]:
    """
    Build ISIN->Name map from /misc CSVs (Trade Republic exports).
    Robust to column naming differences.
    """
    mp: Dict[str, str] = {}
    misc = Path(misc_dir)
    candidates = [
        misc / "TR_Stocks_07_25.csv",
        misc / "TR_ETFs_07_25.csv",
    ]
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            # try semicolon
            df = pd.read_csv(fp, sep=";")

        cols = {c.lower(): c for c in df.columns}

        # find ISIN col
        isin_col = None
        for k in cols:
            if "isin" == k or "isin" in k:
                isin_col = cols[k]
                break
        if isin_col is None:
            continue

        # find name col
        name_col = None
        for key in ["name", "instrument", "bezeichnung", "company", "title", "produkt"]:
            for k in cols:
                if key == k or key in k:
                    name_col = cols[k]
                    break
            if name_col is not None:
                break
        if name_col is None:
            # fallback: first non-isin text col
            for c in df.columns:
                if c == isin_col:
                    continue
                if df[c].dtype == object:
                    name_col = c
                    break

        if name_col is None:
            continue

        for _, row in df[[isin_col, name_col]].dropna().iterrows():
            isin = str(row[isin_col]).strip()
            nm = str(row[name_col]).strip()
            if isin and isin not in mp:
                mp[isin] = nm

    return mp


@torch.no_grad()
def predict_probs_for_asset(
    model: nn.Module,
    X: np.ndarray,
    M: np.ndarray,
    a: int,
    t0: int,
    t1: int,
    lookback: int,
    device: str,
    batch: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns times ts and probabilities p(ts) for anchors t in [t0..t1]
    where t>=lookback-1 and (close mask at t) can be checked outside.
    """
    model.eval()
    A, T, F = X.shape
    assert 0 <= a < A

    ts = np.arange(t0, t1 + 1, dtype=np.int32)
    # build sequences
    seqs = []
    valid_t = []

    for t in ts:
        if t < lookback - 1:
            continue
        x_win = X[a, t - lookback + 1:t + 1, :]  # (L,F)
        m_win = M[a, t - lookback + 1:t + 1, :].astype(np.float32)
        x_cat = np.concatenate([x_win, m_win], axis=1)  # (L,2F)
        seqs.append(torch.from_numpy(x_cat.T).float())  # (2F,L)
        valid_t.append(t)

    if not valid_t:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    probs = np.zeros((len(valid_t),), dtype=np.float32)

    for i in range(0, len(valid_t), batch):
        xb = torch.stack(seqs[i:i + batch], dim=0).to(device)
        logit = model(xb).detach().cpu().numpy().squeeze(1)
        p = 1.0 / (1.0 + np.exp(-logit))
        probs[i:i + len(p)] = p.astype(np.float32)

    return np.array(valid_t, dtype=np.int32), probs


def simulate_long_flat(
    close: np.ndarray,
    ts: np.ndarray,
    probs: np.ndarray,
    thr_enter: float,
    hysteresis: float,
    exec_delay_days: int,
    initial_cash: float,
    trade_fee_eur_roundtrip: float,
) -> Dict[str, object]:
    """
    Simple but realistic per-asset long/flat simulation:
    - signal at day t based on prob(t)
    - execute at day t+delay using Close (conservative vs same-day)
    - long 100% notional with available cash (one position at a time)
    - one completed trade = buy then sell; apply roundtrip fee at exit
    Returns trades list and equity curve.
    """
    assert close.ndim == 1
    n = close.shape[0]

    thr_exit = max(0.0, thr_enter - float(hysteresis))
    delay = int(max(0, exec_delay_days))

    cash = float(initial_cash)
    shares = 0.0
    in_pos = False
    entry_idx = None
    entry_price = None

    equity = np.full((n,), np.nan, dtype=np.float64)
    buys, sells = [], []
    spans = []  # (buy_idx, sell_idx, pnl_frac)

    # decision points: ts corresponds to close indices
    tset = set(int(t) for t in ts.tolist())
    pmap = {int(t): float(p) for t, p in zip(ts.tolist(), probs.tolist())}

    for t in range(n):
        px = float(close[t])
        if px <= 0:
            equity[t] = cash + shares * 0.0
            continue

        # mark-to-market
        equity[t] = cash + shares * px

        if t not in tset:
            continue

        p = pmap[t]

        # determine desired state at decision t (before execution delay)
        want_long = (p >= thr_enter) if not in_pos else (p >= thr_exit)

        # execute at te = t+delay
        te = t + delay
        if te >= n:
            continue
        ex_price = float(close[te])
        if ex_price <= 0:
            continue

        if (not in_pos) and want_long:
            # BUY with all cash
            if cash > 0:
                shares = cash / ex_price
                cash = 0.0
                in_pos = True
                entry_idx = te
                entry_price = ex_price
                buys.append(te)

        elif in_pos and (not want_long):
            # SELL all shares
            proceeds = shares * ex_price
            shares = 0.0
            cash = proceeds

            # apply fixed roundtrip fee on completed trade
            cash = max(0.0, cash - float(trade_fee_eur_roundtrip))

            in_pos = False
            sells.append(te)

            if entry_idx is not None and entry_price is not None and entry_price > 0:
                pnl_frac = (ex_price / entry_price) - 1.0
                spans.append((int(entry_idx), int(te), float(pnl_frac)))
            entry_idx, entry_price = None, None

    # if still in position at end: close at last price (and apply fee)
    if in_pos and shares > 0 and close[-1] > 0:
        proceeds = shares * float(close[-1])
        shares = 0.0
        cash = max(0.0, proceeds - float(trade_fee_eur_roundtrip))
        sells.append(n - 1)
        if entry_idx is not None and entry_price is not None and entry_price > 0:
            pnl_frac = (float(close[-1]) / entry_price) - 1.0
            spans.append((int(entry_idx), int(n - 1), float(pnl_frac)))

    final_equity = float(cash)
    total_return = (final_equity / float(initial_cash)) - 1.0 if initial_cash > 0 else float("nan")
    n_trades = int(min(len(buys), len(sells)))

    # winrate over completed trades
    wins = sum(1 for _, _, pf in spans if pf > 0)
    winrate = (wins / n_trades) if n_trades > 0 else float("nan")

    return {
        "equity": equity,
        "buys": buys,
        "sells": sells,
        "spans": spans,
        "final_equity": final_equity,
        "total_return": float(total_return),
        "n_trades": n_trades,
        "winrate": float(winrate),
    }


def save_trade_plot(
    out_png: str,
    dates: List[str],
    close: np.ndarray,
    buys: List[int],
    sells: List[int],
    spans: List[Tuple[int, int, float]],
    title: str,
) -> None:
    x = np.arange(len(close))
    plt.figure(figsize=(12, 5))
    plt.plot(x, close, linewidth=1.2)

    # shaded holding periods
    for b, s, pnl in spans:
        if b < 0 or s < 0 or b >= len(close) or s >= len(close):
            continue
        color = "green" if pnl > 0 else "red"
        plt.axvspan(b, s, alpha=0.12, color=color)

    # markers
    if buys:
        plt.scatter(buys, close[buys], marker="^")
    if sells:
        plt.scatter(sells, close[sells], marker="v")

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Close")

    # light date ticks
    if len(dates) == len(close) and len(close) > 10:
        step = max(1, len(close) // 8)
        ticks = list(range(0, len(close), step))
        plt.xticks(ticks, [dates[i] for i in ticks], rotation=25, ha="right")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()


def run_test_report(
    model: nn.Module,
    cfg: Config,
    Xn: np.ndarray,
    M: np.ndarray,
    assets: List[str],
    dates: List[str],
    close_raw: np.ndarray,
    close_idx: int,
    te_s: int,
    te_e: int,
    report_base_dir: str,
    misc_dir: str,
    thr: float,
    logger: logging.Logger,
    initial_cash: float,
    trade_fee_eur: float,
    exec_delay_days: int,
    hysteresis: float,
) -> str:
    """
    Creates timestamped test report folder with:
      - summary.csv / summary.json
      - per-asset plots/<ISIN>.png
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(report_base_dir) / f"test_report_{ts}"
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    isin2name = load_isin_name_map(misc_dir)

    A, T, F = Xn.shape
    assert close_raw.shape == (A, T)

    rows = []
    n_kept = 0

    # test slice per asset
    for a in range(A):
        isin = assets[a]
        # require some observed close points in test range
        cm = M[a, te_s:te_e + 1, close_idx].astype(bool)
        if cm.sum() < 50:
            continue

        close = close_raw[a, te_s:te_e + 1].astype(np.float64)
        # optionally set missing closes to NaN for plot cleanliness
        close = close.copy()
        close[~cm] = np.nan

        # predictions only for valid t where lookback window exists and close observed
        # decision range must allow execution delay
        t0 = te_s
        t1 = te_e - max(0, exec_delay_days)
        ts_idx, probs = predict_probs_for_asset(
            model=model,
            X=Xn,
            M=M,
            a=a,
            t0=t0,
            t1=t1,
            lookback=cfg.lookback,
            device=cfg.device,
            batch=2048,
        )

        if len(ts_idx) == 0:
            continue

        # build local close series without leading NaNs for sim: fill NaNs by skipping trades implicitly
        # For simulation, treat NaN prices as invalid => equity stays cash and no exec
        close_sim = np.array(close, dtype=np.float64)
        close_sim[np.isnan(close_sim)] = 0.0

        sim = simulate_long_flat(
            close=close_sim,
            ts=(ts_idx - te_s),          # local index
            probs=probs,
            thr_enter=float(thr),
            hysteresis=float(hysteresis),
            exec_delay_days=int(exec_delay_days),
            initial_cash=float(initial_cash),
            trade_fee_eur_roundtrip=float(trade_fee_eur),
        )

        name = isin2name.get(isin, "")
        title = f"{isin}" + (f" — {name}" if name else "")
        out_png = str(plots_dir / f"{isin}.png")
        save_trade_plot(
            out_png=out_png,
            dates=dates[te_s:te_e + 1],
            close=np.nan_to_num(close, nan=0.0),
            buys=sim["buys"],
            sells=sim["sells"],
            spans=sim["spans"],
            title=title,
        )

        rows.append({
            "isin": isin,
            "name": name,
            "trades": sim["n_trades"],
            "winrate": sim["winrate"],
            "final_equity": sim["final_equity"],
            "total_return_pct": 100.0 * sim["total_return"],
        })
        n_kept += 1

    df = pd.DataFrame(rows).sort_values(by="total_return_pct", ascending=False) if rows else pd.DataFrame(
        columns=["isin", "name", "trades", "winrate", "final_equity", "total_return_pct"]
    )

    summary_csv = outdir / "summary.csv"
    df.to_csv(summary_csv, index=False)

    summary = {
        "n_assets_evaluated": int(n_kept),
        "thr": float(thr),
        "lookback": int(cfg.lookback),
        "horizon": int(cfg.horizon),
        "roundtrip_cost_bps_label": float(cfg.roundtrip_cost_bps),
        "initial_cash": float(initial_cash),
        "trade_fee_eur_roundtrip": float(trade_fee_eur),
        "exec_delay_days": int(exec_delay_days),
        "hysteresis": float(hysteresis),
        "mean_return_pct": float(df["total_return_pct"].mean()) if len(df) else float("nan"),
        "median_return_pct": float(df["total_return_pct"].median()) if len(df) else float("nan"),
        "mean_trades": float(df["trades"].mean()) if len(df) else float("nan"),
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[TESTREPORT] wrote: {str(outdir)}")
    logger.info(f"[TESTREPORT] assets={summary['n_assets_evaluated']} mean_return_pct={summary['mean_return_pct']:.3f}")

    return str(outdir)


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
        scaler: Optional[torch.cuda.amp.GradScaler],
        amp: bool,
        grad_clip: float,
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

        use_amp = bool(amp and device.startswith("cuda") and scaler is not None)

        with amp_autocast(amp_device_type(device), enabled=use_amp):
            logit = model(x)
            logit = torch.clamp(logit, -10.0, 10.0)
            loss_raw = loss_fn(logit, y)
            loss = (loss_raw * v).sum() / (v.sum() + 1e-9)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    # ----------------------------
    # Sweep arguments
    # ----------------------------
    ap.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep (screening mode)")
    ap.add_argument("--trials", type=int, default=48, help="Number of sweep trials")
    ap.add_argument("--sweep_epochs", type=int, default=2, help="Epochs per trial during sweep screening")
    ap.add_argument("--sweep_seed", type=int, default=1234, help="Base RNG seed for sweep sampling")
    ap.add_argument("--sweep_repeats", type=int, default=1,
                    help="Repeat each trial with different model seeds; aggregated ranking (robustness)")
    ap.add_argument("--sweep_seed_stride", type=int, default=1000,
                    help="Stride for deriving per-trial/per-repeat model seeds to avoid collisions")
    ap.add_argument(
        "--resume_sweep_dir",
        default="",
        help=(
            "Resume an interrupted sweep. Provide either the run outdir (containing 'sweep/') "
            "or the sweep folder itself. Already completed (trial,stage,rep) entries in results.jsonl "
            "will be skipped."
        ),
    )
    ap.add_argument("--sweep_two_stage", action="store_true",
                    help="Two-stage sweep: cheap screening first, then train only top-K trials longer/more repeats.")
    ap.add_argument("--sweep_stage1_epochs", type=int, default=1,
                    help="Epochs for stage-1 screening (only if --sweep_two_stage).")
    ap.add_argument("--sweep_stage1_repeats", type=int, default=1,
                    help="Repeats for stage-1 screening (only if --sweep_two_stage).")
    ap.add_argument("--sweep_stage2_topk", type=int, default=20,
                    help="How many trials to keep for stage-2 (only if --sweep_two_stage).")
    ap.add_argument("--sweep_stage2_epochs", type=int, default=3,
                    help="Epochs for stage-2 (only if --sweep_two_stage).")
    ap.add_argument("--sweep_stage2_repeats", type=int, default=3,
                    help="Repeats for stage-2 (only if --sweep_two_stage).")
    # ----------------------------
    # Train-from-trial (load cfg.json)
    # ----------------------------
    ap.add_argument("--cfg", default="", help="Optional: path to a cfg.json from a sweep trial to override Config")
    ap.add_argument("--epochs", type=int, default=0, help="Optional: override cfg.epochs (useful for full training)")

    # Full-train overrides (multi-seed + full(er) validation)
    ap.add_argument("--seed", type=int, default=-1, help="Override cfg.seed (for multi-seed full training runs)")
    ap.add_argument("--max_anchors_train", type=int, default=-1, help = "Override cfg.max_anchors_train (0 disables sampling, -1 keeps cfg)")
    ap.add_argument("--max_anchors_val", type=int, default=-1, help = "Override cfg.max_anchors_val (0 disables sampling, -1 keeps cfg)")
    ap.add_argument("--early_stop_patience", type=int, default=-1, help = "Override cfg.early_stop_patience (0 disables, -1 keeps cfg)")
    ap.add_argument("--early_stop_min_epochs", type=int, default=-1,help = "Override cfg.early_stop_min_epochs (-1 keeps cfg)")
    # ----------------------------
    # TEST / REPORT MODE
    # ----------------------------
    ap.add_argument("--test_only", action="store_true",
                    help="Skip training; load model and run test report only.")
    ap.add_argument("--model_path", default="",
                    help="Path to a trained checkpoint (.pt). If empty, uses --outdir/best_model.pt")
    ap.add_argument("--report_dir", default="",
                    help="Output base dir for test reports. If empty, uses --outdir/test_reports")
    ap.add_argument("--misc_dir", default="",
                    help="Project misc dir containing TR_Stocks_*.csv and TR_ETFs_*.csv. If empty, uses <project_root>/misc")
    ap.add_argument("--test_thr", type=float, default=-1.0,
                    help="Decision threshold for trading. If <0, uses checkpoint val_thr or re-optimizes on val.")
    ap.add_argument("--thr_min", type=float, default=-1.0, help="Override cfg.thr_min (threshold search min)")
    ap.add_argument("--thr_max", type=float, default=-1.0, help="Override cfg.thr_max (threshold search max)")
    ap.add_argument("--min_trades_val", type=int, default=-1,
                    help="Override cfg.min_trades_val (validation min trades)")
    ap.add_argument("--initial_cash", type=float, default=10_000.0,
                    help="Initial cash per-asset simulation (independent).")
    ap.add_argument("--trade_fee_eur", type=float, default=1.0,
                    help="Fixed roundtrip fee in EUR per completed trade (buy+sell).")
    ap.add_argument("--exec_delay_days", type=int, default=1,
                    help="Execution delay in days: signal at t, execute at t+delay (using Close). 0=execute same day.")
    ap.add_argument("--hysteresis", type=float, default=0.02,
                    help="Hysteresis band: enter at thr, exit at thr-hysteresis to reduce churn.")


    args = ap.parse_args()

    cfg = Config()

    # Optional: override Config from a sweep-trial cfg.json (reproducibility)
    if args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            d = json.load(f)

        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Optional: override epochs from CLI (takes precedence over cfg.json)
    if args.epochs and args.epochs > 0:
        cfg.epochs = int(args.epochs)

    # Optional: override seed for multi-seed full training
    if hasattr(args, "seed") and args.seed is not None and int(args.seed) >= 0:
        cfg.seed = int(args.seed)

    # Optional: override sampling anchors (0 disables sampling)
    if hasattr(args, "max_anchors_train") and int(args.max_anchors_train) >= 0:
        cfg.max_anchors_train = int(args.max_anchors_train)
    if hasattr(args, "max_anchors_val") and int(args.max_anchors_val) >= 0:
        cfg.max_anchors_val = int(args.max_anchors_val)

    # Optional: override early stopping knobs
    if hasattr(args, "early_stop_patience") and int(args.early_stop_patience) >= 0:
        cfg.early_stop_patience = int(args.early_stop_patience)
    if hasattr(args, "early_stop_min_epochs") and int(args.early_stop_min_epochs) >= 0:
        cfg.early_stop_min_epochs = int(args.early_stop_min_epochs)

    # Thresholds
    if hasattr(args, "thr_min") and float(args.thr_min) >= 0:
        cfg.thr_min = float(args.thr_min)
    if hasattr(args, "thr_max") and float(args.thr_max) >= 0:
        cfg.thr_max = float(args.thr_max)
    if hasattr(args, "min_trades_val") and int(args.min_trades_val) >= 0:
        cfg.min_trades_val = int(args.min_trades_val)

    set_seed(cfg.seed)
    os.makedirs(args.outdir, exist_ok=True)

    logger = setup_logger(args.outdir)
    logger.info(f"device={cfg.device}")

    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    scaler = AmpGradScaler(amp_device_type(cfg.device), enabled=(cfg.amp and str(cfg.device).startswith("cuda")))


    X = np.load(args.dataset)  # (A,T,F) float32
    M = np.load(args.mask)  # (A,T,F) uint8
    assets = load_text_lines(args.assets)
    dates = load_text_lines(args.dates)
    feat_names = load_json(args.features)

    assert X.shape == M.shape
    A, T, F = X.shape
    safe_log(f"[INFO] loaded cube: A={A}, T={T}, F={F}")

    # --- PATCH: resolve misc/report dirs ---
    project_root = _project_root_from_dataset_path(args.dataset)
    misc_dir = args.misc_dir if args.misc_dir else str(Path(project_root) / "misc")
    report_base = args.report_dir if args.report_dir else str(Path(args.outdir) / "test_reports")

    # Identify Close column
    close_idx = None
    for i, n in enumerate(feat_names):
        if str(n).strip().lower() == "close":
            close_idx = i
            break
    if close_idx is None:
        raise RuntimeError("Could not find 'Close' feature in features json.")

    # ----------------------------
    # SWEEP MODE (screening + optional refit)
    # ----------------------------
    if args.sweep:
        base_cfg = Config()
        base_cfg.device = cfg.device  # keep detected device
        base_cfg.num_workers = cfg.num_workers
        base_cfg.amp = cfg.amp
        base_cfg.cudnn_benchmark = cfg.cudnn_benchmark

        rng = np.random.default_rng(args.sweep_seed)

        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = time_split_indices(T, args.train_frac, args.val_frac)
        splits = ((tr_s, tr_e), (va_s, va_e), (te_s, te_e))

        # --- NEW: resume support ---
        # If --resume_sweep_dir is provided, it can point to:
        #   - the run outdir (contains "sweep/") OR
        #   - the sweep folder itself
        if args.resume_sweep_dir:
            r = Path(args.resume_sweep_dir).resolve()
            if r.name.lower() == "sweep":
                sweep_outdir = str(r)
            else:
                sweep_outdir = str(r / "sweep")
        else:
            sweep_outdir = os.path.join(args.outdir, "sweep")

        os.makedirs(sweep_outdir, exist_ok=True)
        results_path = os.path.join(sweep_outdir, "results.jsonl")

        completed_keys, existing_rows = load_existing_sweep_results(results_path)
        if args.resume_sweep_dir:
            logger.info(
                f"[SWEEP][RESUME] sweep_outdir={sweep_outdir} "
                f"existing_entries={len(completed_keys)}"
            )

        safe_log(f"[INFO] SWEEP start: trials={args.trials} sweep_epochs={args.sweep_epochs} out={sweep_outdir}")

        leaderboard = []

        def _run_repeats_for(cfg_base: Config, trial: int, tdir: str, repeats: int, epochs: int, stage: str,
                             completed_keys: set | None = None, existing_rows: dict | None = None):
            rep_metrics = []
            rep_times = []
            if completed_keys is None:
                completed_keys = set()
            if existing_rows is None:
                existing_rows = {}
            for rep in range(int(repeats)):
                k = (int(trial), str(stage), int(rep))
                if k in completed_keys and k in existing_rows:
                    # reuse existing metrics; do NOT append duplicate jsonl rows
                    row0 = existing_rows[k]
                    metrics0 = {
                        "best_val_avg_rnet": float(row0.get("best_val_avg_rnet", float("nan"))),
                        "best_val_thr": float(row0.get("best_val_thr", float("nan"))),
                        "best_val_trades": int(row0.get("best_val_trades", 0)),
                        "best_val_winrate": float(row0.get("best_val_winrate", float("nan"))),
                        "best_val_score": float(row0.get("best_val_score", float("nan"))),
                        "best_val_bce": float(row0.get("best_val_bce", float("nan"))),
                        "best_val_acc": float(row0.get("best_val_acc", float("nan"))),
                    }
                    rep_metrics.append(metrics0)
                    rep_times.append(float(row0.get("time_s", float("nan"))))
                    continue

                rep_cfg = Config(**asdict(cfg_base))
                rep_cfg.epochs = int(epochs)
                rep_cfg.seed = int(
                    args.sweep_seed + trial * args.sweep_seed_stride + rep + (0 if stage == "stage1" else 100_000))

                rep_out = os.path.join(tdir, f"{stage}_rep_{rep:02d}")
                os.makedirs(rep_out, exist_ok=True)

                t0 = time.time()
                try:
                    metrics = run_single_training(
                        cfg=rep_cfg, args=args, X=X, M=M,
                        assets=assets, dates=dates, feat_names=feat_names,
                        close_idx=close_idx, splits=splits, outdir=rep_out
                    )
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                        # mark trial as failed but keep sweep running
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        metrics = {
                            "best_val_avg_rnet": float("nan"),
                            "best_val_thr": float("nan"),
                            "best_val_trades": 0,
                            "best_val_bce": float("nan"),
                            "best_val_acc": float("nan"),
                        }
                        logger.exception(f"[SWEEP] OOM in trial={trial} rep={rep} stage={stage} -> skipping")
                    else:
                        raise

                dt = time.time() - t0
                rep_times.append(dt)

                row_rep = {
                    "trial": trial,
                    "rep": rep,
                    "stage": stage,
                    "time_s": round(dt, 2),
                    **asdict(rep_cfg),
                    **metrics,
                }
                write_jsonl(results_path, row_rep)
                completed_keys.add(k)
                existing_rows[k] = row_rep
                rep_metrics.append(metrics)

            return rep_metrics, rep_times

        # ---------- Stage 1: cheap screening ----------
        stage1_records = []  # (score, trial, trial_cfg, tdir)
        for trial in range(1, args.trials + 1):
            tdir = make_trial_outdir(sweep_outdir, trial)

            # If cfg.json exists, always use it (keeps backward compatibility for already-sampled trials)
            cfg_path = os.path.join(tdir, "cfg.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                trial_cfg = Config(**d)
            else:
                # Deterministic per-trial RNG: allows resume even if process restarts mid-sweep
                rng_trial = np.random.default_rng(int(args.sweep_seed + trial))
                trial_cfg = sample_trial_cfg(base_cfg, rng_trial)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(trial_cfg), f, indent=2)

            trial_cfg.seed_trials = int(args.sweep_seed + trial)

            if args.sweep_two_stage:
                rep_metrics, rep_times = _run_repeats_for(
                    cfg_base=trial_cfg,
                    trial=trial,
                    tdir=tdir,
                    repeats=args.sweep_stage1_repeats,
                    epochs=args.sweep_stage1_epochs,
                    stage="stage1",
                    completed_keys=completed_keys,
                    existing_rows=existing_rows,
                )
            else:
                # legacy behavior: directly run full sweep settings
                rep_metrics, rep_times = _run_repeats_for(
                    cfg_base=trial_cfg,
                    trial=trial,
                    tdir=tdir,
                    repeats=args.sweep_repeats,
                    epochs=args.sweep_epochs,
                    stage="stage_full",
                    completed_keys=completed_keys,
                    existing_rows=existing_rows,
                )

            vals = np.array([m["best_val_avg_rnet"] for m in rep_metrics], dtype=np.float64)
            score = float(np.nanmedian(vals)) if np.isfinite(np.nanmedian(vals)) else float("-inf")
            stage1_records.append((score, trial, trial_cfg, tdir))

            if not args.sweep_two_stage:
                # also write leaderboard row now (single-stage)
                trades = np.array([m["best_val_trades"] for m in rep_metrics], dtype=np.float64)
                bce = np.array([m["best_val_bce"] for m in rep_metrics], dtype=np.float64)
                acc = np.array([m["best_val_acc"] for m in rep_metrics], dtype=np.float64)
                agg = {
                    "best_val_avg_rnet_med": float(np.nanmedian(vals)),
                    "best_val_avg_rnet_mean": float(np.nanmean(vals)),
                    "best_val_trades_med": int(np.nanmedian(trades)),
                    "best_val_bce_mean": float(np.nanmean(bce)),
                    "best_val_acc_mean": float(np.nanmean(acc)),
                    "time_s_mean": float(np.mean(rep_times)) if rep_times else float("nan"),
                    "sweep_repeats": int(args.sweep_repeats),
                }
                row = {"trial": trial, "time_s": round(agg["time_s_mean"], 2), **asdict(trial_cfg), **agg}
                leaderboard.append(row)

        # ---------- Stage 2: intensify only top-K ----------
        if args.sweep_two_stage:
            stage1_records.sort(key=lambda x: x[0], reverse=True)
            topk = stage1_records[: int(args.sweep_stage2_topk)]
            logger.info(f"[SWEEP] stage1 done -> stage2 on topK={len(topk)} (of {len(stage1_records)})")

            for rank_i, (score1, trial, trial_cfg, tdir) in enumerate(topk, start=1):
                rep_metrics, rep_times = _run_repeats_for(
                    cfg_base=trial_cfg,
                    trial=trial,
                    tdir=tdir,
                    repeats=args.sweep_stage2_repeats,
                    epochs=args.sweep_stage2_epochs,
                    stage="stage2",
                )

                vals = np.array([m["best_val_avg_rnet"] for m in rep_metrics], dtype=np.float64)
                trades = np.array([m["best_val_trades"] for m in rep_metrics], dtype=np.float64)
                bce = np.array([m["best_val_bce"] for m in rep_metrics], dtype=np.float64)
                acc = np.array([m["best_val_acc"] for m in rep_metrics], dtype=np.float64)

                agg = {
                    "best_val_avg_rnet_med": float(np.nanmedian(vals)),
                    "best_val_avg_rnet_mean": float(np.nanmean(vals)),
                    "best_val_trades_med": int(np.nanmedian(trades)),
                    "best_val_bce_mean": float(np.nanmean(bce)),
                    "best_val_acc_mean": float(np.nanmean(acc)),
                    "time_s_mean": float(np.mean(rep_times)) if rep_times else float("nan"),
                    "sweep_repeats": int(args.sweep_stage2_repeats),
                    "stage1_score_med": float(score1),
                    "stage2_rank": int(rank_i),
                }

                row = {"trial": trial, "time_s": round(agg["time_s_mean"], 2), **asdict(trial_cfg), **agg}
                leaderboard.append(row)

            safe_log(
                f"[SWEEP] trial={trial:04d} model={trial_cfg.model_type} "
                f"avg_rnet_med={row['best_val_avg_rnet_med']:.6f} avg_rnet_mean={row['best_val_avg_rnet_mean']:.6f} "
                f"trades_med={row['best_val_trades_med']} repeats={row['sweep_repeats']} "
                f"L={trial_cfg.lookback} H={trial_cfg.horizon} theta={trial_cfg.ignore_band_logret:.3f} "
                f"lr={trial_cfg.lr:.1e} wd={trial_cfg.weight_decay:.1e} "
                f"do={trial_cfg.dropout:.2f} tcn_do={trial_cfg.tcn_dropout:.2f} "
                f"tcn(ch={trial_cfg.tcn_channels},lv={trial_cfg.tcn_levels},k={trial_cfg.tcn_kernel_size}) "
                f"cost={trial_cfg.roundtrip_cost_bps:.0f}bps time~={row['time_s']:.1f}s"
            )

        # sort by robust metric first (median), then trades
        leaderboard.sort(key=lambda r: (r["best_val_avg_rnet_med"], r["best_val_trades_med"]), reverse=True)

        top_path = os.path.join(sweep_outdir, "leaderboard_top.csv")
        with open(top_path, "w", encoding="utf-8") as f:
            cols = [
                "trial", "model_type",
                "best_val_avg_rnet_med", "best_val_avg_rnet_mean", "best_val_trades_med", "sweep_repeats",
                "lookback", "horizon", "ignore_band_logret", "lr", "weight_decay", "dropout",
                "roundtrip_cost_bps", "batch_size", "max_anchors_train", "max_anchors_val",
                "best_val_bce_mean", "best_val_acc_mean", "time_s"
            ]
            f.write(",".join(cols) + "\n")
            for r in leaderboard[:50]:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

        safe_log(f"[INFO] SWEEP done. Top list written: {top_path}")
        safe_log("[DONE]")
        return

    # ----------------------------
    # TEST ONLY MODE
    # ----------------------------
    if args.test_only:
        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = time_split_indices(T, args.train_frac, args.val_frac)
        safe_log(f"[INFO] time split: train=[{tr_s},{tr_e}] val=[{va_s},{va_e}] test=[{te_s},{te_e}]")

        close_raw = X[:, :, close_idx].copy()

        # normalization with train-only stats (same as training path)
        X_train = X[:, tr_s:tr_e + 1, :]
        M_train = M[:, tr_s:tr_e + 1, :]
        mu, sig = masked_mean_std(X_train, M_train)
        Xn = apply_masked_standardize(X, M, mu, sig)

        # model path
        model_path = args.model_path if args.model_path else os.path.join(args.outdir, "best_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        ckpt = torch.load(model_path, map_location=cfg.device)

        # allow cfg override from checkpoint (important for lookback/horizon/model_type)
        if isinstance(ckpt, dict) and "cfg" in ckpt:
            for k, v in ckpt["cfg"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        model = build_model(cfg, in_ch=2 * F).to(cfg.device)
        model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)

        # threshold choice
        thr = float(args.test_thr)
        if thr < 0:
            # prefer stored threshold if present
            if isinstance(ckpt, dict) and "val_thr" in ckpt:
                thr = float(ckpt["val_thr"])
            else:
                thr = 0.85  # fallback

        logger.info(f"[TESTONLY] model={model_path} thr={thr:.3f} misc_dir={misc_dir}")

        run_test_report(
            model=model,
            cfg=cfg,
            Xn=Xn,
            M=M,
            assets=assets,
            dates=dates,
            close_raw=close_raw,
            close_idx=close_idx,
            te_s=te_s,
            te_e=te_e,
            report_base_dir=report_base,
            misc_dir=misc_dir,
            thr=thr,
            logger=logger,
            initial_cash=args.initial_cash,
            trade_fee_eur=args.trade_fee_eur,
            exec_delay_days=args.exec_delay_days,
            hysteresis=args.hysteresis,
        )

        safe_log("[DONE]")
        return

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
        theta=cfg.ignore_band_logret,
        max_anchors=cfg.max_anchors_train,
        seed=cfg.seed_trials,
    )
    val_ds = CubeProfitDataset(
        X=X, M=M, close_idx=close_idx,
        t_start=va_s, t_end=va_e,
        lookback=cfg.lookback, horizon=cfg.horizon,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        asset_filter=asset_keep,
        require_close_obs=True,
        close_raw=close_raw,
        theta=cfg.ignore_band_logret,
        max_anchors=cfg.max_anchors_val,
        seed=cfg.seed_trials + 1,
    )
    test_ds = CubeProfitDataset(
        X=X, M=M, close_idx=close_idx,
        t_start=te_s, t_end=te_e,
        lookback=cfg.lookback, horizon=cfg.horizon,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        asset_filter=asset_keep,
        require_close_obs=True,
        close_raw=close_raw,
        theta=cfg.ignore_band_logret,
        max_anchors=0,
        seed=cfg.seed_trials + 2,
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
    model = build_model(cfg, in_ch=in_ch).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val_avg_rnet = -float("inf")
    best_path = os.path.join(args.outdir, "best_model.pt")
    saved_any_ckpt = False
    no_improve = 0

    logger.info("training start")
    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        improved = False

        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            opt=opt,
            device=cfg.device,
            logger=logger,
            epoch=epoch,
            scaler=scaler,
            amp=cfg.amp,
            grad_clip=cfg.grad_clip,
            log_every=100,
        )

        val_metrics = evaluate_classifier(model, val_loader, cfg.device)

        # --- NEW: profit-oriented validation metrics (do NOT rely on BCE/acc for trading) ---
        thresholds = np.linspace(cfg.thr_min, cfg.thr_max, cfg.threshold_grid, dtype=np.float32)
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

        # append trial row (epoch-level)
        row = {
            "epoch": epoch,
            "lookback": cfg.lookback,
            "horizon": cfg.horizon,
            "theta": cfg.ignore_band_logret,
            "lr": cfg.lr,
            "wd": cfg.weight_decay,
            "dropout": cfg.dropout,
            "cost_bps": cfg.roundtrip_cost_bps,
            "val_thr": val_thr,
            "val_trades": val_trades,
            "val_avg_rnet": val_avg_rnet,
            "val_winrate": val_winrate,
            "val_bce": val_metrics["bce"],
            "val_acc": val_metrics["acc@0.5"],
        }
        with open(os.path.join(args.outdir, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        ep_time = time.time() - ep_t0
        logger.info(
            f"E{epoch:03d} DONE "
            f"train_loss={tr_loss:.6f} "
            f"val_bce={val_metrics['bce']:.6f} val_acc@0.5={val_metrics['acc@0.5']:.4f} "
            f"n_val={val_metrics['n']} "
            f"epoch_time_s={ep_time:.1f}"
        )

        # Checkpoint by profitability proxy, not BCE ---
        if np.isfinite(val_avg_rnet) and val_trades >= cfg.min_trades_val and val_avg_rnet > best_val_avg_rnet:
            best_val_avg_rnet = val_avg_rnet

            ckpt_obj = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_thr": float(val_thr),
                "val_info": val_info,
                "cfg": asdict(cfg),
            }

            tmp_path = best_path + ".tmp"
            torch.save(ckpt_obj, tmp_path)
            os.replace(tmp_path, best_path)  # atomic on POSIX/Windows
            saved_any_ckpt = True

            logger.info(
                f"[CKPT] saved {os.path.basename(best_path)} "
                f"(best_val_avg_rnet={best_val_avg_rnet:.6f}, trades={val_trades}, thr={val_thr:.3f})"
            )

        # --- Early stopping on validation trading metric (best_val_avg_rnet) ---
        if cfg.early_stop_patience and cfg.early_stop_patience > 0:
            if improved:
                no_improve = 0
            else:
                no_improve += 1

            if epoch >= int(cfg.early_stop_min_epochs) and no_improve >= int(cfg.early_stop_patience):
                logger.info(
                    f"[EARLY STOP] epoch={epoch} no_improve={no_improve} "
                    f"best_val_avg_rnet={best_val_avg_rnet:.6f}"
                )
                break

        sched.step()

        if not saved_any_ckpt:
            logger.warning(
                f"[CKPT] no checkpoint met profitability gate (min_trades_val={cfg.min_trades_val}). "
                f"Saving last-epoch model as fallback to avoid missing best_model.pt."
            )
            ckpt_obj = {
                "epoch": cfg.epochs,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_thr": float(val_thr),
                "val_info": val_info,
                "cfg": asdict(cfg),
            }
            tmp_path = best_path + ".tmp"
            torch.save(ckpt_obj, tmp_path)
            os.replace(tmp_path, best_path)
            saved_any_ckpt = True

    safe_log(f"[INFO] best model saved: {best_path} (best_val_avg_rnet={best_val_avg_rnet:.6f})")

    # Reload best
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])

    # Threshold optimization to maximize profit probability among taken trades
    thresholds = np.linspace(cfg.thr_min, cfg.thr_max, cfg.threshold_grid, dtype=np.float32)
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

    # If script is started with CLI args (terminal), don't overwrite them.
    # Only inject IDE defaults when no args were provided.
    if len(sys.argv) == 1:
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


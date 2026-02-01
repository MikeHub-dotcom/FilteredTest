#!/usr/bin/env python3
"""
Batch launcher for full-training runs (trial cfg.json + multi-seed), replacing the cmd/bat approach.

What it does:
- Iterates over a list of sweep trial IDs and seeds
- Builds cfg.json path: <base>/sweep/trial_XXXX/cfg.json
- Creates an output directory per run: runs_full_trial{trial}_seed{seed}
- Runs `python main.py ...` as a subprocess
- Streams stdout+stderr to <outdir>/train.log
- Stops on first failure (default), or continues if requested

Run (inside the same venv you use for training):
    python run_fulltrain_batch.py

Optional overrides:
    python run_fulltrain_batch.py --base runs_tcn_sweep_robust --trials 107,76,36,111 --seeds 20260121,20261121,20262121 --epochs 120
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional


def parse_int_list(csv: str) -> List[int]:
    csv = (csv or "").strip()
    if not csv:
        return []
    out: List[int] = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_one(
    python_exe: str,
    project_root: Path,
    base: Path,
    dataset: Path,
    mask: Path,
    assets: Path,
    dates: Path,
    features: Path,
    trial_id: int,
    seed: int,
    epochs: int,
    max_anchors_val: int,
    early_stop_min_epochs: int,
    early_stop_patience: int,
    variant: str,
    sidecar_ids: Optional[Path],
    extra_args: List[str],
    stop_on_error: bool,
) -> int:
    tid4 = f"{trial_id:04d}"
    cfg_path = base / "sweep" / f"trial_{tid4}" / "cfg.json"
    outdir = (base / f"runs_full_trial{trial_id}_seed{seed}_{variant}").resolve()
    ensure_dir(outdir)

    log_path = outdir / "train.log"
    main_py = project_root / "main.py"

    cmd = [
        python_exe, str(main_py),
        "--dataset", str(dataset),
        "--mask", str(mask),
        "--assets", str(assets),
        "--dates", str(dates),
        "--features", str(features),
        "--cfg", str(cfg_path),
        "--outdir", str(outdir),
        "--epochs", str(int(epochs)),
        "--seed", str(int(seed)),
        "--max_anchors_val", str(int(max_anchors_val)),
        "--early_stop_min_epochs", str(int(early_stop_min_epochs)),
        "--early_stop_patience", str(int(early_stop_patience)),
    ] + list(extra_args or [])

    # Sidecar variant: append flags expected by main.py
    if variant == "sidecars":
        if sidecar_ids is None:
            raise RuntimeError("variant=sidecars requires --sidecar_ids <path-to-meta_ids_clean_*.npy>")
        cmd += ["--use_sidecars", "--sidecar_ids", str(sidecar_ids)]

    print(f"[RUN] trial={trial_id} tid={tid4} seed={seed}")
    print(f"      cfg={cfg_path}")
    print(f"      out={outdir}")
    print(f"      log={log_path}")

    # Pre-flight checks
    if not main_py.exists():
        raise FileNotFoundError(f"main.py not found at: {main_py}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg.json not found at: {cfg_path}")
    for p, name in [(dataset, "dataset"), (mask, "mask"), (assets, "assets"), (dates, "dates"), (features, "features")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} file not found at: {p}")

    with open(log_path, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] START trial={trial_id} seed={seed}\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=os.environ.copy(),
        )
        rc = proc.wait()

        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] END rc={rc}\n")
        f.flush()

    if rc != 0:
        print(f"[FAIL] trial={trial_id} seed={seed} rc={rc} (see {log_path})", file=sys.stderr)
        return rc if stop_on_error else 0

    print(f"[OK] trial={trial_id} seed={seed}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs_tcn_sweep_robust")
    ap.add_argument("--dataset", default="dataset/dataset_2026-02-01_masked.npy")
    ap.add_argument("--mask", default="dataset/mask_2026-02-01_masked.npy")
    ap.add_argument("--assets", default="dataset/assets_2026-02-01.txt")
    ap.add_argument("--dates", default="dataset/dates_2026-02-01.txt")
    ap.add_argument("--features", default="dataset/features_2026-02-01.json")

    ap.add_argument("--seeds", default="20260121,20261121,20262121")
    ap.add_argument("--epochs", type=int, default=120)

    ap.add_argument("--trials", default="107")

    ap.add_argument("--mode", choices=["screening", "full"], default="screening",
                    help="screening: 2 seeds, shorter training; full: 5 seeds, longer training.")
    ap.add_argument("--variants", default="baseline,sidecars",
                    help="Comma-list of run variants: baseline,sidecars")
    ap.add_argument("--sidecar_ids", default="",
                    help="Path to meta_ids_clean_*.npy (A x I). Required if 'sidecars' variant is enabled.")

    ap.add_argument("--max_anchors_val", type=int, default=0)
    ap.add_argument("--early_stop_min_epochs", type=int, default=30)
    ap.add_argument("--early_stop_patience", type=int, default=12)

    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--extra", default="", help="Extra args appended verbatim (single string), e.g. \"--device cuda --amp\"")

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    base = (project_root / args.base).resolve()
    dataset = (project_root / args.dataset).resolve()
    mask = (project_root / args.mask).resolve()
    assets = (project_root / args.assets).resolve()
    dates = (project_root / args.dates).resolve()
    features = (project_root / args.features).resolve()

    # Mode presets
    if args.mode == "full":
        # If user did not override seeds, switch to 5-seed default
        if args.seeds.strip() == "20260121,20261121":
            args.seeds = "20260121,20261121,20262121,20263121,20264121"
        # If user did not override epochs, use a longer default
        if int(args.epochs) == 50:
            args.epochs = 120
        # more conservative early stop for full training
        if int(args.early_stop_min_epochs) == 10:
            args.early_stop_min_epochs = 30
        if int(args.early_stop_patience) == 6:
            args.early_stop_patience = 12

    trials = parse_int_list(args.trials)
    seeds = parse_int_list(args.seeds)
    if not trials:
        raise ValueError("No trials provided")
    if not seeds:
        raise ValueError("No seeds provided")

    variants = [v.strip() for v in (args.variants or "").split(",") if v.strip()]
    if not variants:
        raise ValueError("No variants provided (expected e.g. baseline,sidecars)")
    for v in variants:
        if v not in ("baseline", "sidecars"):
            raise ValueError(f"Unknown variant '{v}'. Use baseline or sidecars.")

    sidecar_ids = Path(args.sidecar_ids).resolve() if args.sidecar_ids.strip() else None

    stop_on_error = not args.continue_on_error
    extra_args = args.extra.strip().split() if args.extra.strip() else []

    print("[BATCH] starting")
    print(f"        base={base}")
    print(f"        trials={trials}")
    print(f"        seeds={seeds}")
    print(f"        python={args.python}")
    print(f"        epochs={args.epochs}")
    if extra_args:
        print(f"        extra_args={extra_args}")
    print(f"        mode={args.mode}")
    print(f"        variants={variants}")
    if "sidecars" in variants:
        print(f"        sidecar_ids={sidecar_ids}")

    for trial_id in trials:
        for seed in seeds:
            for variant in variants:
                rc = run_one(
                    python_exe=args.python,
                    project_root=project_root,
                    base=base,
                    dataset=dataset,
                    mask=mask,
                    assets=assets,
                    dates=dates,
                    features=features,
                    trial_id=trial_id,
                    seed=seed,
                    epochs=int(args.epochs),
                    max_anchors_val=int(args.max_anchors_val),
                    early_stop_min_epochs=int(args.early_stop_min_epochs),
                    early_stop_patience=int(args.early_stop_patience),
                    variant=variant,
                    sidecar_ids=sidecar_ids,
                    extra_args=extra_args,
                    stop_on_error=stop_on_error,
                )
                if rc != 0 and stop_on_error:
                    sys.exit(rc)

    print("[BATCH] done")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


KS_P_THRESHOLD = 0.05
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "Data" / "normalized" / "events_d01.csv"
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"


def load_rows(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    totals: dict[str, int] = defaultdict(int)
    minutes: list[int] = []
    counts: list[int] = []
    execs: list[float] = []
    fids: list[str] = []
    with path.open("r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = row["function_id"]
            c = int(row["count"])
            e = float(row["avg_exec_time_ms"])
            m = int(row["minute"])
            totals[fid] += c
            fids.append(fid)
            minutes.append(m)
            counts.append(c)
            execs.append(e)
    sorted_fids = [f for f, _ in sorted(totals.items(), key=lambda kv: -kv[1])]
    return sorted_fids, np.asarray(minutes, dtype=np.int32), np.asarray(counts, dtype=np.int64), np.asarray(execs, dtype=np.float64), np.asarray(fids)


def build_training_arrays(
    sorted_fids: list[str],
    minute_arr: np.ndarray,
    count_arr: np.ndarray,
    exec_arr: np.ndarray,
    fid_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rank_of = {fid: i for i, fid in enumerate(sorted_fids)}
    n_funcs = max(1, len(sorted_fids) - 1)
    rank_norm = np.array([rank_of[f] / n_funcs for f in fid_arr], dtype=np.float32)
    hour = minute_arr.astype(np.float32) / 60.0
    hs = np.sin(2.0 * np.pi * hour / 24.0)
    hc = np.cos(2.0 * np.pi * hour / 24.0)
    C = np.stack([rank_norm, hs.astype(np.float32), hc.astype(np.float32)], axis=1)

    rng = np.random.default_rng(0)
    exec_jitter = rng.uniform(-0.49, 0.49, size=exec_arr.shape)
    count_jitter = rng.uniform(-0.49, 0.49, size=count_arr.shape)
    X_raw = np.stack(
        [
            np.log1p(np.clip(count_arr.astype(np.float64) + count_jitter, 0.0, None)),
            np.log1p(np.clip(exec_arr + exec_jitter, 0.0, None)),
        ],
        axis=1,
    )
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    X = ((X_raw - mean) / std).astype(np.float32)

    meta = {
        "x_mean": mean.tolist(),
        "x_std": std.tolist(),
        "n_training_rows": int(X.shape[0]),
        "n_functions_trained": len(sorted_fids),
    }
    return X, C, meta


def ks_gate(real_X_raw: np.ndarray, sampled_X_raw: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ks_2samp

    p_count = ks_2samp(real_X_raw[:, 0], sampled_X_raw[:, 0]).pvalue
    p_exec = ks_2samp(real_X_raw[:, 1], sampled_X_raw[:, 1]).pvalue
    return float(p_count), float(p_exec)


def joint_metric(real_X_raw: np.ndarray, sampled_X_raw: np.ndarray) -> float:
    from scipy.stats import spearmanr

    corr_real, _ = spearmanr(real_X_raw[:, 0], real_X_raw[:, 1])
    corr_sampled, _ = spearmanr(sampled_X_raw[:, 0], sampled_X_raw[:, 1])
    return float(abs(corr_real - corr_sampled))


def train(
    data_path: Path,
    n_epochs: int,
    sample_size: int,
    n_layers: int,
    hidden: tuple[int, ...],
    seed: int,
    output_dir: Path = WEIGHTS_DIR,
    weights_name: str = "flow_v1.pt",
    meta_name: str = "flow_v1_meta.json",
) -> dict:
    import torch

    from probaforms.models import RealNVP

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[train_flow] loading {data_path}")
    sorted_fids, minute_arr, count_arr, exec_arr, fid_arr = load_rows(data_path)
    print(f"[train_flow] rows={len(minute_arr)} unique_functions={len(sorted_fids)}")

    X, C, meta = build_training_arrays(sorted_fids, minute_arr, count_arr, exec_arr, fid_arr)

    perm = np.random.default_rng(seed).permutation(X.shape[0])
    cut = int(0.9 * X.shape[0])
    train_idx, test_idx = perm[:cut], perm[cut:]

    print(f"[train_flow] training RealNVP: layers={n_layers} hidden={hidden} epochs={n_epochs}")
    model = RealNVP(
        n_layers=n_layers,
        hidden=hidden,
        n_epochs=n_epochs,
        batch_size=512,
        lr=1e-3,
    )
    model.fit(X[train_idx], C[train_idx])

    test_C = C[test_idx[:sample_size]]
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        sampled_norm = model.sample(test_C)
    if hasattr(sampled_norm, "detach"):
        sampled_norm = sampled_norm.detach().cpu().numpy()
    sampled_norm = np.asarray(sampled_norm, dtype=np.float32)

    mean = np.asarray(meta["x_mean"])
    std = np.asarray(meta["x_std"])
    sampled_raw = sampled_norm * std + mean
    real_raw = X[test_idx[:sample_size]] * std + mean

    p_count, p_exec = ks_gate(real_raw, sampled_raw)
    spearman_diff = joint_metric(real_raw, sampled_raw)
    print(f"[train_flow] KS p-values: count={p_count:.4f} exec={p_exec:.4f}  threshold={KS_P_THRESHOLD}")
    print(f"[train_flow] Joint metric (Spearman diff): {spearman_diff:.4f}")

    passed = (p_count > KS_P_THRESHOLD) and (p_exec > KS_P_THRESHOLD)
    meta.update(
        {
            "ks_p_count": p_count,
            "ks_p_exec": p_exec,
            "spearman_diff": spearman_diff,
            "ks_threshold": KS_P_THRESHOLD,
            "passed": passed,
            "function_id_order": sorted_fids,
            "n_layers": n_layers,
            "hidden": list(hidden),
            "cond_dim": 3,
            "x_dim": 2,
        }
    )

    if passed:
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / weights_name
        meta_path = output_dir / meta_name
        torch.save(model.state_dict(), weights_path)
        meta_path.write_text(json.dumps(meta, indent=2))
        size_mb = weights_path.stat().st_size / 1_048_576
        print(f"[train_flow] saved {weights_path} ({size_mb:.2f} MB)")
        print(f"[train_flow] saved {meta_path}")
    else:
        print("[train_flow] KS-gate FAILED; weights not saved.")

    return meta


def main() -> int:
    p = argparse.ArgumentParser(description="Train conditional RealNVP on an aggregate trace")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=WEIGHTS_DIR)
    p.add_argument("--weights-name", type=str, default="flow_v1.pt")
    p.add_argument("--meta-name", type=str, default="flow_v1_meta.json")
    p.add_argument("--verify-only", action="store_true", help="load saved weights, re-run KS gate only")
    args = p.parse_args()

    if args.verify_only:
        raise SystemExit("verify-only mode not implemented yet; retrain with --n-epochs N")

    meta = train(
        data_path=args.data,
        n_epochs=args.n_epochs,
        sample_size=args.sample_size,
        n_layers=args.n_layers,
        hidden=tuple(args.hidden),
        seed=args.seed,
        output_dir=args.output_dir,
        weights_name=args.weights_name,
        meta_name=args.meta_name,
    )
    print("[train_flow] RESULT " + json.dumps({
        "passed": bool(meta["passed"]),
        "ks_p_count": float(meta["ks_p_count"]),
        "ks_p_exec": float(meta["ks_p_exec"]),
        "ks_threshold": float(meta["ks_threshold"]),
    }))
    return 0 if meta["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

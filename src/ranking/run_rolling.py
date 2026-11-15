"""Rolling-origin evaluation (quarterly retraining, expanding window).

Usage: python -m src.ranking.run_rolling [--folds 1,2,...] [--protocols B,A]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from .common import OUT, PROTOCOL_A, PROTOCOL_B, build_features, daily_sums, feature_list, latest, load_panel, ranking_metrics, split
from .models import Scorer

FOLDS = [  # (train_end = test_start, test_end)
    ("2025-03-01", "2025-06-01"),
    ("2025-06-01", "2025-09-01"),
    ("2025-09-01", "2025-12-01"),
    ("2025-12-01", "2026-03-01"),
    ("2026-03-01", "2026-06-01"),
    ("2026-06-01", "2026-09-01"),
]
FEATURES = OUT / "features.parquet"


def log(msg):
    print(msg, flush=True)


def get_features():
    if FEATURES.exists():
        df = pd.read_parquet(FEATURES)
        sets = json.loads((OUT / "feature_sets.json").read_text())
        return df, sets
    t0 = time.time()
    panel = load_panel()
    df, sets = build_features(panel)
    df.to_parquet(FEATURES, index=False)
    (OUT / "feature_sets.json").write_text(json.dumps(sets, indent=1))
    log(f"features built in {time.time() - t0:.0f}s: {df.shape}")
    return df, sets


def regression_row(name, y, pred, base):
    return {"model": name, "mae": mean_absolute_error(y, pred), "rmse": mean_squared_error(y, pred) ** 0.5,
            "r2": r2_score(y, pred), "mae_pers": mean_absolute_error(y, base),
            "rmse_pers": mean_squared_error(y, base) ** 0.5, "r2_pers": r2_score(y, base),
            "skill_mae": 1 - mean_absolute_error(y, pred) / mean_absolute_error(y, base)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    ap.add_argument("--protocols", default="B,A")
    args = ap.parse_args()
    df, sets = get_features()
    protocols = {"A": PROTOCOL_A, "B": PROTOCOL_B}
    for fi in [int(f) for f in args.folds.split(",")]:
        test_start, test_end = FOLDS[fi - 1]
        train, valid, test = split(df, test_start, test_start, test_end)
        for pname in args.protocols.split(","):
            prot = protocols[pname]
            tag = f"fold{fi}_{pname}"
            if (OUT / f"rolling_{tag}_metrics.csv").exists():
                log(f"skip {tag}")
                continue
            log(f"== {tag}: train {len(train)} valid {len(valid)} test {len(test)} ({test_start} to {test_end})")
            num, cat = feature_list(sets, prot)
            sc = Scorer(train, valid, test, num, cat, prot, log=log)
            scores = sc.baselines()
            scores.update(sc.hgb_scores())
            if pname == "B":
                scores["FQI"] = sc.fitted_q()
                scores.update(sc.lightgbm_scores())
            # ranking metrics
            rows, daily = [], {}
            for name, s in scores.items():
                for k in (5, 10, 20):
                    for h in (1, 6):
                        rows.append({"fold": fi, "protocol": pname, "policy": name, **ranking_metrics(test, s, k, h)})
                    d = daily_sums(test, s, 10, 1)
                    daily[(name, k)] = d
            met = pd.DataFrame(rows)
            met.to_csv(OUT / f"rolling_{tag}_metrics.csv", index=False)
            # daily sums for bootstrap (k=10, h=1) and per-instant scores for later analyses
            dd = pd.concat({n: daily_sums(test, s, 10, 1) for n, s in scores.items()}, names=["policy"])
            dd.to_csv(OUT / f"rolling_{tag}_daily_k10.csv")
            keep = test[["time", "plant", "state", "subsystem", "y_loss", "y_rate", "y_event", "y_reason", "loss", "loss_lag1"]
                        + [f"y_loss_h{h}" for h in range(1, 7)]].copy()
            for n, s in scores.items():
                keep["s_" + n] = s
            keep.to_parquet(OUT / f"rolling_{tag}_scores.parquet", index=False)
            # point forecast quality (rate and event)
            base = latest(test, "rate", prot).fillna(0).to_numpy()
            reg = regression_row("HGB-rate", test["y_rate"].to_numpy(), sc.rate_pred, base)
            ev = test["y_event"].to_numpy().astype(int)
            pe = scores["HGB-event"] / np.maximum(latest(test, "ref", prot).fillna(0).to_numpy(), 1e-9)
            reg.update({"auc_event": roc_auc_score(ev, pe), "ap_event": average_precision_score(ev, pe), "prev_event": ev.mean(),
                        "fold": fi, "protocol": pname, "n_train": len(train) + len(valid), "n_test": len(test)})
            pd.DataFrame([reg]).to_csv(OUT / f"rolling_{tag}_forecast.csv", index=False)
            pd.Series(sc.times).to_csv(OUT / f"rolling_{tag}_fit_seconds.csv")
            k10 = met[(met.k == 10) & (met.horizon == 1)].sort_values("oracle_eff", ascending=False)
            log(k10[["policy", "capture", "oracle_eff", "ndcg_at_k"]].to_string(index=False))


if __name__ == "__main__":
    main()

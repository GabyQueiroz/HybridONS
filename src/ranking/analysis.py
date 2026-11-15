"""Post-hoc analyses on the saved scores of the rolling-origin folds (protocol B unless stated).

Outputs (results_ranking/):
  rolling_summary.csv           pooled metrics per policy, k, horizon (all 6 folds = 18 test months)
  rolling_bootstrap_k10.csv     day-block bootstrap CI of oracle efficiency and paired difference vs persistence
  rolling_monthly_k10.csv       oracle efficiency per test month
  rolling_subsystem_k10.csv     capture per subsystem
  rolling_onsets_k10.csv        detection of restriction onsets and sharp increases
  rolling_turnover_k10.csv      top-k overlap with the previous interval and with the oracle
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .common import OUT, bootstrap_ci, cumulative_loss, ranking_metrics, topk_frame

POLICIES = ["Random", "Persistence", "Recent mean", "Previous day", "Plant-hour mean", "Plant size", "HGB-event",
            "HGB-rate", "HGB-loss", "HGB-return", "FQI", "LGBM-loss", "LambdaMART"]


def load(protocol="B"):
    files = sorted(glob.glob(str(OUT / f"rolling_fold*_{protocol}_scores.parquet")))
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        d["fold"] = int(f.split("fold")[1].split("_")[0])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def pooled(df, policies):
    rows = []
    for name in policies:
        for k in (5, 10, 20):
            for h in (1, 6):
                rows.append({"policy": name, **ranking_metrics(df, df["s_" + name].to_numpy(), k, h)})
    return pd.DataFrame(rows)


def daily(df, name, k=10, h=1):
    times = df["time"].to_numpy()
    gain = df["y_loss"].to_numpy() if h == 1 else cumulative_loss(df, h)
    sel = topk_frame(times, df["s_" + name].to_numpy(), k)
    ora = topk_frame(times, gain, k)
    day = df["time"].dt.floor("D").to_numpy()
    return pd.DataFrame({"day": day, "cap": np.where(sel, gain, 0.0), "ora": np.where(ora, gain, 0.0)}).groupby("day").sum()


def main():
    df = load("B")
    df = df.sort_values(["time", "plant"]).reset_index(drop=True)
    pols = [p for p in POLICIES if "s_" + p in df.columns]
    pooled(df, pols).to_csv(OUT / "rolling_summary.csv", index=False)

    for h in (1, 6):
        boot = bootstrap_ci({p: daily(df, p, 10, h) for p in pols}, ref="Persistence")
        boot["horizon"] = h
        boot.to_csv(OUT / f"rolling_bootstrap_k10_h{h}.csv", index=False)
        # paired comparison against the best simple baseline as well
        boot2 = bootstrap_ci({p: daily(df, p, 10, h) for p in pols}, ref="HGB-loss")
        boot2.to_csv(OUT / f"rolling_bootstrap_vs_hgbloss_k10_h{h}.csv", index=False)

    # monthly and subsystem breakdown
    df["month"] = df["time"].dt.to_period("M").astype(str)
    rows = []
    for m, g in df.groupby("month"):
        for p in pols:
            r = ranking_metrics(g, g["s_" + p].to_numpy(), 10, 1)
            rows.append({"month": m, "policy": p, **r})
    pd.DataFrame(rows).to_csv(OUT / "rolling_monthly_k10.csv", index=False)

    times = df["time"].to_numpy()
    gain = df["y_loss"].to_numpy()
    sub_rows = []
    for p in pols:
        sel = topk_frame(times, df["s_" + p].to_numpy(), 10)
        for s, g in df.groupby("subsystem"):
            idx = g.index.to_numpy()
            sub_rows.append({"policy": p, "subsystem": s, "share_of_loss": gain[idx].sum() / gain.sum(),
                             "captured_share_within_subsystem": gain[idx][sel[idx]].sum() / gain[idx].sum()})
    pd.DataFrame(sub_rows).to_csv(OUT / "rolling_subsystem_k10.csv", index=False)

    # onsets: plant without loss at the latest available interval (t-1) and with loss at t+1
    last = df["loss_lag1"].fillna(0).to_numpy()
    onset = (last <= 0.5) & (gain > 0.5)
    jump = (gain - last > 50)                                  # increase above 50 MWmed between t-1 and t+1
    on_rows = []
    for p in pols:
        sel = topk_frame(times, df["s_" + p].to_numpy(), 10)
        on_rows.append({"policy": p, "onset_events": int(onset.sum()), "onset_selected_share": float(sel[onset].mean()),
                        "onset_loss_captured_share": float(gain[onset & sel].sum() / gain[onset].sum()),
                        "jump_events": int(jump.sum()), "jump_selected_share": float(sel[jump].mean()),
                        "jump_loss_captured_share": float(gain[jump & sel].sum() / gain[jump].sum()),
                        "onset_share_of_total_loss": float(gain[onset].sum() / gain.sum())})
    pd.DataFrame(on_rows).to_csv(OUT / "rolling_onsets_k10.csv", index=False)

    # turnover: how many of the top-10 change between consecutive intervals; overlap with oracle
    tv_rows = []
    oracle_sel = topk_frame(times, gain, 10)
    key = df[["time", "plant"]].copy()
    for p in pols + ["Oracle"]:
        sel = oracle_sel if p == "Oracle" else topk_frame(times, df["s_" + p].to_numpy(), 10)
        chosen = key[sel].copy()
        chosen["prev"] = chosen["time"] - pd.Timedelta(minutes=30)
        sets = chosen.groupby("time")["plant"].apply(frozenset)
        prev = sets.reindex(sets.index - pd.Timedelta(minutes=30))
        same = [len(a & b) / 10 for a, b in zip(sets.to_numpy(), prev.to_numpy()) if isinstance(b, frozenset)]
        ov = (sel & oracle_sel).sum() / oracle_sel.sum()
        tv_rows.append({"policy": p, "mean_overlap_with_previous_interval": float(np.mean(same)),
                        "share_intervals_identical_set": float(np.mean(np.isclose(same, 1.0))),
                        "overlap_with_oracle_set": float(ov)})
    pd.DataFrame(tv_rows).to_csv(OUT / "rolling_turnover_k10.csv", index=False)

    # concentration of loss among plants
    plant_loss = df.groupby("plant")["y_loss"].sum().sort_values(ascending=False)
    conc = pd.DataFrame({"top_n": [5, 10, 15, 20], "share": [plant_loss.head(n).sum() / plant_loss.sum() for n in (5, 10, 15, 20)]})
    conc.to_csv(OUT / "rolling_concentration.csv", index=False)

    # protocol A vs B (policies present in both)
    try:
        a = load("A").sort_values(["time", "plant"]).reset_index(drop=True)
        rows = []
        for prot, d in (("A", a), ("B", df)):
            for p in ["Persistence", "HGB-rate", "HGB-loss", "HGB-return", "HGB-event"]:
                r = ranking_metrics(d, d["s_" + p].to_numpy(), 10, 1)
                rows.append({"protocol": prot, "policy": p, **r})
        pd.DataFrame(rows).to_csv(OUT / "rolling_protocols_k10.csv", index=False)
    except ValueError:
        pass
    print(pd.read_csv(OUT / "rolling_bootstrap_k10_h1.csv").to_string())


if __name__ == "__main__":
    main()

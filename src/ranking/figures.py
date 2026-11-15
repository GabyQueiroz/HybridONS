"""Vector figures for the manuscript (PDF, TrueType fonts, sized for a 7-inch text width)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import OUT, load_panel

FIG = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Arial", "font.size": 8, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
COL1, COL2 = 3.45, 7.0
MAIN = ["HGB-loss", "LGBM-loss", "FQI", "HGB-return", "LambdaMART", "HGB-rate", "Persistence", "Previous day",
        "HGB-event", "Plant size", "Plant-hour mean", "Recent mean", "Random"]
COLORS = {"HGB-loss": "#1b6ca8", "LGBM-loss": "#5aa0d0", "FQI": "#8c564b", "HGB-return": "#c49c94", "LambdaMART": "#9467bd",
          "HGB-rate": "#2ca02c", "Persistence": "#d62728", "Previous day": "#ff9896", "HGB-event": "#bcbd22",
          "Plant size": "#7f7f7f", "Plant-hour mean": "#c7c7c7", "Recent mean": "#e377c2", "Random": "#17becf"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_audit():
    p = load_panel()
    d = p[p["ref"] > 0].copy()
    d["month"] = d["time"].dt.to_period("M").dt.to_timestamp()
    d["av"] = (d["ref"] - d["avail"]).clip(lower=0)
    d["res"] = (np.minimum(d["ref"], d["avail"]) - d["gen"]).clip(lower=0)
    d["coded_b"] = d["reason"] != ""
    m = d.groupby(["month", "coded_b"])["loss"].sum().unstack(fill_value=0) * 0.5 / 1000
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(len(m))
    ax.bar(x, m[True], color="#1b6ca8", label="Coded records")
    ax.bar(x, m[False], bottom=m[True], color="#f2a541", label="Records without code")
    ax.set_xticks(x[::3])
    ax.set_xticklabels(m.index.strftime("%b %Y")[::3], rotation=45, ha="right")
    ax.set_ylabel("Derived loss (GWh)")
    ax.legend(loc="upper left")
    ax.set_title("(a) Monthly derived loss", fontsize=8.5, loc="left")
    ax = axes[1]
    hour = d.groupby(d["time"].dt.hour + d["time"].dt.minute / 60)[["res", "av"]].sum() * 0.5 / 1000
    ax.stackplot(hour.index, hour["res"], hour["av"], colors=["#1b6ca8", "#bdbdbd"], labels=["Restriction component", "Availability component"])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Derived loss (GWh)")
    ax.set_xlim(5, 19.5)
    ax.set_ylim(0, hour.sum(axis=1).max() * 1.35)
    ax.legend(loc="upper right")
    ax.set_title("(b) Intraday profile", fontsize=8.5, loc="left")
    fig.tight_layout()
    save(fig, "fig_audit")
    stats = {
        "share_uncoded": float(d.loc[~d["coded_b"], "loss"].sum() / d["loss"].sum()),
        "share_availability": float(d["av"].sum() / d["loss"].sum()),
        "share_uncoded_availability": float(d.loc[~d["coded_b"], "av"].sum() / d.loc[~d["coded_b"], "loss"].sum()),
        "total_gwh": float(d["loss"].sum() * 0.5 / 1000),
    }
    by_reason = d.groupby("reason").agg(records=("loss", "size"), plants=("plant", "nunique"), loss_gwh=("loss", "sum"),
                                         av_gwh=("av", "sum"))
    by_reason[["loss_gwh", "av_gwh"]] *= 0.5 / 1000
    by_reason.to_csv(OUT / "audit_by_reason.csv")
    pd.Series(stats).to_csv(OUT / "audit_stats.csv")


def fig_monthly():
    m = pd.read_csv(OUT / "rolling_monthly_k10.csv")
    pols = ["HGB-loss", "FQI", "LambdaMART", "HGB-rate", "Persistence", "HGB-event"]
    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    for p in pols:
        g = m[m["policy"] == p]
        ax.plot(pd.to_datetime(g["month"]), g["oracle_eff"], marker="o", ms=3, lw=1.2, color=COLORS[p], label=p)
    ax.set_ylabel("Oracle efficiency ($k=10$)")
    ax.set_xlabel("Test month")
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.legend(ncol=3, loc="lower left")
    save(fig, "fig_monthly")


def fig_bootstrap():
    b = pd.read_csv(OUT / "rolling_bootstrap_k10_h1.csv")
    b = b[b["policy"] != "Random"].sort_values("eff")
    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    y = np.arange(len(b))
    ax.errorbar(b["eff"], y, xerr=[b["eff"] - b["eff_lo"], b["eff_hi"] - b["eff"]], fmt="o", ms=3.5, color="#1b6ca8",
                ecolor="#555555", elinewidth=0.8, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(b["policy"])
    ax.set_xlabel("Oracle efficiency, $k=10$ (95% bootstrap interval)")
    ax.grid(axis="x", lw=0.3, alpha=0.6)
    save(fig, "fig_bootstrap")


def fig_fqi_horizon():
    f = pd.read_csv(OUT / "main_fqi.csv")
    h = pd.read_csv(OUT / "main_horizon.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    ax = axes[0]
    it = f[f["iterations"] != "direct return"].copy()
    it["iterations"] = it["iterations"].astype(int)
    d = f[f["iterations"] == "direct return"].iloc[0]
    ax.plot(it["iterations"], it["eff_h1"], marker="o", ms=3.5, color="#8c564b", label="FQI, gain over 1 interval")
    ax.plot(it["iterations"], it["eff_h6"], marker="s", ms=3.5, color="#1b6ca8", label="FQI, gain over 6 intervals")
    ax.axhline(d["eff_h1"], ls="--", lw=0.9, color="#8c564b", label="Direct $R^{(6)}$ regression, 1 interval")
    ax.axhline(d["eff_h6"], ls=":", lw=0.9, color="#1b6ca8", label="Direct $R^{(6)}$ regression, 6 intervals")
    ax.set_xlabel("Fitted-Q iterations $J$")
    ax.set_ylabel("Oracle efficiency ($k=10$)")
    ax.set_ylim(0.855, 0.945)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(a) Fitted-Q iterations", fontsize=8.5, loc="left")
    ax = axes[1]
    for gamma, g in h[h["eval_horizon"] == h["train_horizon"]].groupby("gamma"):
        ax.plot(g["train_horizon"], g["oracle_eff"], marker="o", ms=3.5, label=f"$\\gamma$={gamma}")
    ax.set_xlabel("Horizon $H$ (training and evaluation)")
    ax.set_ylabel("Oracle efficiency ($k=10$)")
    ax.set_xticks([1, 3, 6, 12])
    ax.legend()
    ax.set_title("(b) Return horizon and discount", fontsize=8.5, loc="left")
    fig.tight_layout()
    save(fig, "fig_fqi_horizon")


def fig_families():
    r = pd.read_csv(OUT / "main_families_regression.csv")
    k = pd.read_csv(OUT / "main_families_ranking.csv")
    k["model"] = k["policy"].str.replace(r" s\d+$", "", regex=True)
    agg = k.groupby("model")["oracle_eff"].agg(["mean", "std", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    y = np.arange(len(agg))
    ax.barh(y, agg["mean"], xerr=agg["std"].fillna(0), color="#5aa0d0", ecolor="#333333", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} (n={int(c)})" for m, c in zip(agg.index, agg["count"])])
    lo = max(0.0, agg["mean"].min() - 0.05)
    ax.set_xlim(lo, min(1.0, agg["mean"].max() + 0.02))
    ax.set_xlabel("Oracle efficiency ($k=10$), mean $\\pm$ SD over seeds")
    save(fig, "fig_families")


def fig_learning_curves():
    hst = pd.read_csv(OUT / "main_families_nn_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.1), sharey=False)
    for ax, model in zip(axes, ["MLP", "GRU"]):
        for s, g in hst[hst["model"] == model].groupby("seed"):
            ax.plot(g["epoch"], g["valid_mae_mw"] * 0.5, marker="o", ms=2.5, lw=1, label=f"seed {s}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE (MWh)")
        ax.set_title(model, fontsize=8.5, loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_learning_curves")


if __name__ == "__main__":
    for fn in [fig_audit, fig_monthly, fig_bootstrap, fig_fqi_horizon, fig_families, fig_learning_curves]:
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("missing input for", fn.__name__, e)

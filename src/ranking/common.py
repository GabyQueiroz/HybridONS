"""Data preparation, decision protocols and evaluation.

Main changes with respect to src/hybridons_experiments.py (earlier version):
- lags are computed on a regular 30-min grid per plant (gaps no longer shift the lags);
- two information protocols: A (data up to t) and B (one-interval reporting delay, data up to t-1);
- persistence uses the latest loss available under each protocol;
- candidate set defined by the target time window, not by quantities observed at t+1;
- losses reported as energy (MWh = MWmed x 0.5 h);
- training rows whose 6-step targets cross into the test period are purged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "restricao_coff_fotovoltaica_consolidado.parquet"
OUT = ROOT / "results_ranking"
OUT.mkdir(exist_ok=True)

STEP = pd.Timedelta(minutes=30)
LAGS = [1, 2, 6, 12, 47, 48, 96]          # 47 = same time of the target interval on the previous day
GAMMA = 0.92
HORIZON = 6
DAY_START, DAY_END = 6.0, 18.5            # target interval window (local time, hours)
CATEGORICAL = ["plant", "state", "subsystem"]


@dataclass
class Protocol:
    name: str
    current: bool          # True: quantities at t are available when ranking for t+1


PROTOCOL_A = Protocol("A", current=True)
PROTOCOL_B = Protocol("B", current=False)


def load_panel(path: Path = DATA) -> pd.DataFrame:
    cols = ["id_ons", "id_estado", "id_subsistema", "din_instante", "val_geracao", "val_geracaolimitada",
            "val_disponibilidade", "val_geracaoreferencia", "cod_razaorestricao", "cod_origemrestricao"]
    raw = pd.read_parquet(path, columns=cols)
    raw = raw.rename(columns={"id_ons": "plant", "id_estado": "state", "id_subsistema": "subsystem",
                              "din_instante": "time", "val_geracao": "gen", "val_geracaolimitada": "lim",
                              "val_disponibilidade": "avail", "val_geracaoreferencia": "ref",
                              "cod_razaorestricao": "reason", "cod_origemrestricao": "origin"})
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.drop_duplicates(["plant", "time"]).sort_values(["plant", "time"])

    # regular grid per plant (first to last record), so that shift(h) = h intervals
    grids = []
    for plant, g in raw.groupby("plant", sort=False):
        idx = pd.date_range(g["time"].iloc[0], g["time"].iloc[-1], freq=STEP)
        grids.append(pd.DataFrame({"plant": plant, "time": idx}))
    panel = pd.concat(grids, ignore_index=True).merge(raw, on=["plant", "time"], how="left")
    panel["observed"] = panel["gen"].notna()
    for c in ["state", "subsystem"]:
        panel[c] = panel.groupby("plant")[c].transform(lambda s: s.ffill().bfill())
    panel["reason"] = panel["reason"].fillna("").astype(str).str.strip()
    panel["origin"] = panel["origin"].fillna("").astype(str).str.strip()
    panel["coded"] = (panel["reason"] != "").astype(float)
    panel["loss"] = (panel["ref"] - panel["gen"]).clip(lower=0)                     # MWmed in the interval
    panel["rate"] = np.where(panel["ref"] > 0, panel["loss"] / panel["ref"], 0.0)
    panel["unavail"] = (panel["ref"] - panel["avail"]).clip(lower=0)
    panel.loc[~panel["observed"], ["coded", "rate"]] = np.nan
    return panel


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    p = panel
    g = p.groupby("plant", sort=False)
    feats = {}
    for col in ["loss", "rate", "coded", "gen", "ref", "avail"]:
        for h in LAGS:
            feats[f"{col}_lag{h}"] = g[col].shift(h)
    feats["reason_lag1"] = g["reason"].shift(1)
    # targets
    feats["y_loss"] = g["loss"].shift(-1)
    feats["y_rate"] = g["rate"].shift(-1)
    feats["y_event"] = g["coded"].shift(-1)
    feats["y_ref"] = g["ref"].shift(-1)
    feats["y_reason"] = g["reason"].shift(-1)
    fut = [g["loss"].shift(-h) for h in range(1, 13)]
    for h in range(1, 13):
        feats[f"y_loss_h{h}"] = fut[h - 1]
    feats["next_obs"] = g["observed"].shift(-1)
    df = pd.concat([p.reset_index(drop=True), pd.DataFrame(feats)], axis=1)

    tt = df["time"] + STEP                                                   # target interval start
    hour = tt.dt.hour + tt.dt.minute / 60
    df["t_target"] = tt
    df["hour"] = hour
    df["dow"] = tt.dt.dayofweek
    df["month"] = tt.dt.month
    df["doy_sin"] = np.sin(2 * np.pi * tt.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * tt.dt.dayofyear / 365.25)
    df["origin_code"] = df["origin"]
    df["reason_code"] = df["reason"]

    keep = (df["next_obs"] == True) & (hour >= DAY_START) & (hour <= DAY_END) & df["loss_lag96"].notna()  # noqa: E712
    df = df[keep].copy()
    df["y_loss"] = df["y_loss"].fillna(0.0)
    df["y_rate"] = df["y_rate"].fillna(0.0)
    df["y_event"] = df["y_event"].fillna(0.0)
    for h in range(1, 13):
        df[f"y_loss_h{h}"] = df[f"y_loss_h{h}"].fillna(0.0)
    df = df.sort_values(["time", "plant"]).reset_index(drop=True)
    for c in CATEGORICAL + ["reason_code", "origin_code", "reason_lag1"]:
        df[c] = df[c].fillna("").astype(str)

    calendar = ["hour", "dow", "month", "doy_sin", "doy_cos"]
    lagged = [f"{c}_lag{h}" for c in ["loss", "rate", "coded", "gen", "ref", "avail"] for h in LAGS]
    current = ["gen", "lim", "avail", "ref", "loss", "rate", "coded"]
    sets = {
        "calendar": calendar,
        "lagged": lagged,
        "current": current,
        "codes_current": ["reason_code", "origin_code"],
        "codes_lagged": ["reason_lag1"] + [f"coded_lag{h}" for h in LAGS],
        "id": CATEGORICAL,
    }
    return df, sets


def feature_list(sets: dict[str, list[str]], protocol: Protocol, drop: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names for a protocol and an ablation."""
    num = list(sets["calendar"]) + list(sets["lagged"])
    cat = list(sets["id"]) + ["reason_lag1"]
    if protocol.current:
        num += sets["current"]
        cat += sets["codes_current"]
    if "lags" in drop:
        num = [c for c in num if "_lag" not in c or c.endswith("_lag1")]          # keep only the latest lag
    if "id" in drop:
        cat = [c for c in cat if c not in sets["id"]]
    if "codes" in drop:
        num = [c for c in num if not c.startswith("coded")]
        cat = [c for c in cat if c not in ("reason_code", "origin_code", "reason_lag1")]
    if "calendar" in drop:
        num = [c for c in num if c not in sets["calendar"]]
    return num, cat


def latest(df: pd.DataFrame, col: str, protocol: Protocol) -> pd.Series:
    """Latest value of a quantity available at decision time."""
    return df[col] if protocol.current else df[f"{col}_lag1"]


def discounted_return(df: pd.DataFrame, gamma: float = GAMMA, horizon: int = HORIZON) -> np.ndarray:
    return sum((gamma ** (h - 1)) * df[f"y_loss_h{h}"].to_numpy() for h in range(1, horizon + 1))


def cumulative_loss(df: pd.DataFrame, horizon: int) -> np.ndarray:
    return sum(df[f"y_loss_h{h}"].to_numpy() for h in range(1, horizon + 1))


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame, train_end: str, test_start: str, test_end: str, valid_months: int = 1):
    """Expanding-window split. The last `valid_months` before train_end are used for validation.
    Rows whose 12-step future crosses the next period are purged."""
    train_end = pd.Timestamp(train_end)
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    valid_start = train_end - pd.DateOffset(months=valid_months)
    purge = 12 * STEP
    train = df[df["time"] < valid_start - purge]
    valid = df[(df["time"] >= valid_start) & (df["time"] < train_end - purge)]
    test = df[(df["time"] >= test_start) & (df["time"] < test_end)]
    return train, valid, test


# ---------------------------------------------------------------------------
# ranking evaluation
# ---------------------------------------------------------------------------
def topk_frame(times: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the top-k rows per instant (ties broken by row order)."""
    order = np.lexsort((-score, times))
    t_sorted = times[order]
    first = np.r_[True, t_sorted[1:] != t_sorted[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank = np.arange(len(order)) - group_start
    mask = np.zeros(len(order), dtype=bool)
    mask[order[rank < k]] = True
    return mask


def ndcg_at_k(times: np.ndarray, score: np.ndarray, gain: np.ndarray, k: int) -> float:
    def dcg(s):
        order = np.lexsort((-s, times))
        t_sorted = times[order]
        first = np.r_[True, t_sorted[1:] != t_sorted[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
        rank = np.arange(len(order)) - start
        keep = rank < k
        contrib = gain[order][keep] / np.log2(rank[keep] + 2)
        return pd.Series(contrib).groupby(t_sorted[keep]).sum()
    d, i = dcg(score), dcg(gain)
    valid = i > 0
    return float((d[valid] / i[valid]).mean())


def ranking_metrics(test: pd.DataFrame, score: np.ndarray, k: int, horizon: int = 1, pos_threshold: float = 0.5) -> dict:
    times = test["time"].to_numpy()
    gain = test["y_loss"].to_numpy() if horizon == 1 else cumulative_loss(test, horizon)
    sel = topk_frame(times, score, k)
    ora = topk_frame(times, gain, k)
    pos = gain > pos_threshold * horizon
    captured = gain[sel].sum()
    oracle = gain[ora].sum()
    total = gain.sum()
    n_pos_t = pd.Series(pos).groupby(times).sum()
    sel_pos_t = pd.Series(pos & sel).groupby(times).sum()
    return {
        "k": k,
        "horizon": horizon,
        "capture": captured / total,
        "oracle_eff": captured / oracle,
        "precision_at_k": float((pos & sel).sum() / sel.sum()),
        "recall_at_k": float(sel_pos_t[n_pos_t > 0].sum() / n_pos_t[n_pos_t > 0].sum()),
        "ndcg_at_k": ndcg_at_k(times, score, gain, k),
        "captured_mwh": captured * 0.5,
        "oracle_mwh": oracle * 0.5,
        "total_mwh": total * 0.5,
    }


def daily_sums(test: pd.DataFrame, score: np.ndarray, k: int, horizon: int = 1) -> pd.DataFrame:
    times = test["time"].to_numpy()
    gain = test["y_loss"].to_numpy() if horizon == 1 else cumulative_loss(test, horizon)
    sel = topk_frame(times, score, k)
    ora = topk_frame(times, gain, k)
    day = test["time"].dt.floor("D").to_numpy()
    return pd.DataFrame({"day": day, "cap": np.where(sel, gain, 0.0), "ora": np.where(ora, gain, 0.0)}).groupby("day").sum()


def bootstrap_ci(daily: dict[str, pd.DataFrame], ref: str | None = None, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Day-block bootstrap of oracle efficiency (and paired difference to `ref`)."""
    rng = np.random.default_rng(seed)
    days = next(iter(daily.values())).index
    n = len(days)
    idx = rng.integers(0, n, size=(reps, n))
    rows = []
    ref_eff = None
    if ref is not None:
        r = daily[ref].loc[days]
        ref_eff = r["cap"].to_numpy()[idx].sum(1) / r["ora"].to_numpy()[idx].sum(1)
    for name, d in daily.items():
        d = d.loc[days]
        eff = d["cap"].to_numpy()[idx].sum(1) / d["ora"].to_numpy()[idx].sum(1)
        row = {"policy": name, "eff": d["cap"].sum() / d["ora"].sum(),
               "eff_lo": np.quantile(eff, 0.025), "eff_hi": np.quantile(eff, 0.975)}
        if ref_eff is not None and name != ref:
            diff = eff - ref_eff
            row.update({"diff_vs_" + ref: float(np.mean(eff) - np.mean(ref_eff)),
                        "diff_lo": np.quantile(diff, 0.025), "diff_hi": np.quantile(diff, 0.975),
                        "p_le_0": float((diff <= 0).mean())})
        rows.append(row)
    return pd.DataFrame(rows)

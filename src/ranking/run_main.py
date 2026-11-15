"""Experiments on the original split (train 2024-04..2026-03, validation 2026-04..06, test 2026-07..08).

Stages (run in order, each writes results_ranking/main_<stage>.csv):
  grid      HGB hyperparameter search on the validation months
  families  model families for next-interval loss (HGB, LightGBM x5 seeds, XGBoost, CatBoost, MLP x5, GRU x3)
  policies  all ranking policies under protocols A and B
  ablation  feature-group ablations (protocol B, HGB-loss)
  horizon   discount factor and horizon sensitivity of the return model
  fqi       fitted-Q iterations versus direct return regression
  coded     analyses with and without restriction code
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .common import (OUT, PROTOCOL_A, PROTOCOL_B, cumulative_loss, daily_sums, discounted_return, feature_list,
                     latest, load_panel, ranking_metrics)
from .models import HGB_PARAMS, Encoder, Scorer, _cat_fix, hgb_regressor
from .run_rolling import get_features

TRAIN_END, VALID_END, TEST_END = "2026-04-01", "2026-07-01", "2026-09-01"


def log(msg):
    print(msg, flush=True)


def main_split(df):
    purge = pd.Timedelta(hours=6)
    train = df[df["time"] < pd.Timestamp(TRAIN_END) - purge]
    valid = df[(df["time"] >= TRAIN_END) & (df["time"] < pd.Timestamp(VALID_END) - purge)]
    test = df[(df["time"] >= VALID_END) & (df["time"] < TEST_END)]
    return train, valid, test


def rank_rows(test, scores, tag, ks=(5, 10, 20), horizons=(1, 6)):
    rows = []
    for name, s in scores.items():
        for k in ks:
            for h in horizons:
                rows.append({"tag": tag, "policy": name, **ranking_metrics(test, s, k, h)})
    return rows


def reg_row(name, y, p, extra=None):
    r = {"model": name, "mae_mwh": mean_absolute_error(y, p) * 0.5, "rmse_mwh": mean_squared_error(y, p) ** 0.5 * 0.5}
    r.update(extra or {})
    return r


# ---------------------------------------------------------------------------
def stage_grid(df, sets):
    train, valid, _ = main_split(df)
    num, cat = feature_list(sets, PROTOCOL_B)
    enc = Encoder(cat).fit(train)
    xtr, xva = _cat_fix(enc.transform(train, num), len(num)), _cat_fix(enc.transform(valid, num), len(num))
    rows = []
    for leaves in (31, 63, 127):
        for iters in (150, 300, 500):
            t0 = time.perf_counter()
            m = hgb_regressor(num, cat, max_leaf_nodes=leaves, max_iter=iters).fit(xtr, train["y_loss"])
            p = np.clip(m.predict(xva), 0, None)
            met = ranking_metrics(valid, p, 10, 1)
            rows.append({"max_leaf_nodes": leaves, "max_iter": iters, "valid_mae_mwh": mean_absolute_error(valid["y_loss"], p) * 0.5,
                         "valid_oracle_eff_k10": met["oracle_eff"], "valid_ndcg_k10": met["ndcg_at_k"], "seconds": time.perf_counter() - t0})
            log(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "main_grid.csv", index=False)


# ---------------------------------------------------------------------------
class TabMLP(torch.nn.Module):
    def __init__(self, n_num, cards, hidden=256, dropout=0.1):
        super().__init__()
        self.emb = torch.nn.ModuleList([torch.nn.Embedding(c + 1, min(16, (c + 1) // 2 + 1)) for c in cards])
        d = n_num + sum(e.embedding_dim for e in self.emb)
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden // 2, 1))

    def forward(self, xn, xc, seq=None):
        z = [xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)]
        return self.net(torch.cat(z, 1)).squeeze(1)


class SeqGRU(torch.nn.Module):
    def __init__(self, n_seq, n_num, cards, hidden=64):
        super().__init__()
        self.gru = torch.nn.GRU(n_seq, hidden, batch_first=True)
        self.head = TabMLP(n_num + hidden, cards, hidden=128, dropout=0.1)

    def forward(self, xn, xc, seq):
        _, h = self.gru(seq)
        return self.head(torch.cat([xn, h[-1]], 1), xc)


def nn_arrays(train, valid, test, num, cat):
    mu, sd = train[num].mean(), train[num].std().replace(0, 1)

    def xn(d):
        return torch.tensor(((d[num] - mu) / sd).fillna(0).to_numpy(np.float32))
    maps = {c: {v: i + 1 for i, v in enumerate(sorted(train[c].unique()))} for c in cat}

    def xc(d):
        return torch.tensor(np.stack([d[c].map(maps[c]).fillna(0).to_numpy(np.int64) for c in cat], 1))
    cards = [len(maps[c]) for c in cat]
    return (xn(train), xc(train)), (xn(valid), xc(valid)), (xn(test), xc(test)), cards


def sequences(panel_pos, values, rows_pos, offset, length=12):
    """Past window of `length` steps ending at row position + offset (offset 0 = t, -1 = t-1)."""
    idx = rows_pos[:, None] + offset + np.arange(-length + 1, 1)[None, :]
    seq = values[np.clip(idx, 0, len(values) - 1)]
    return np.nan_to_num(seq, nan=0.0).astype(np.float32)


def train_nn(model, tr, va, ytr, yva, seqs=None, epochs=30, patience=4, lr=1e-3, seed=0, batch=2048):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scale = 100.0
    ytr_t = torch.tensor(ytr / scale, dtype=torch.float32)
    best, best_state, wait, hist = np.inf, None, 0, []
    s_tr, s_va = (seqs or (None, None))[:2]

    def predict(x, s):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x[0]), 16384):
                out.append(model(x[0][i:i + 16384], x[1][i:i + 16384], None if s is None else s[i:i + 16384]))
        return torch.cat(out).numpy() * scale

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr_t))
        tl = []
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            pred = model(tr[0][b], tr[1][b], None if s_tr is None else s_tr[b])
            loss = torch.nn.functional.huber_loss(pred, ytr_t[b], delta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())
        pv = np.clip(predict(va, s_va), 0, None)
        mae = mean_absolute_error(yva, pv)
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(tl)), "valid_mae_mw": mae})
        log(f"      epoch {ep + 1} train {np.mean(tl):.4f} valid MAE {mae:.3f}")
        if mae < best - 1e-3:
            best, wait = mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, predict, hist


def stage_families(df, sets, skip_gru=False):
    import catboost
    import lightgbm as lgb
    import xgboost as xgb
    train, valid, test = main_split(df)
    num, cat = feature_list(sets, PROTOCOL_B)
    enc = Encoder(cat).fit(pd.concat([train, valid]))
    fit = pd.concat([train, valid])
    xf, xt = _cat_fix(enc.transform(fit, num), len(num)), _cat_fix(enc.transform(test, num), len(num))
    cat_idx = list(range(len(num), len(num) + len(cat)))
    y, yt = fit["y_loss"].to_numpy(), test["y_loss"].to_numpy()
    scores, regs, timing, hist_rows = {}, [], {}, []

    def add(name, pred, secs, seed=None):
        pred = np.clip(pred, 0, None)
        scores[name if seed is None else f"{name} s{seed}"] = pred
        regs.append(reg_row(name, yt, pred, {"seed": seed, "fit_seconds": secs}))
        log(f"  {name} seed={seed}: MAE {regs[-1]['mae_mwh']:.3f} MWh ({secs:.0f}s)")

    t0 = time.perf_counter()
    m = hgb_regressor(num, cat).fit(xf, y)
    add("HGB", m.predict(xt), time.perf_counter() - t0)
    t1 = time.perf_counter()
    m.predict(xt[: test["plant"].nunique()])
    timing["HGB_inference_one_instant_ms"] = (time.perf_counter() - t1) * 1000

    for s in (1, 2, 3, 4, 5):
        t0 = time.perf_counter()
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=s, verbose=-1)
        m.fit(xf, y, categorical_feature=cat_idx)
        add("LightGBM", m.predict(xt), time.perf_counter() - t0, s)

    t0 = time.perf_counter()
    xf_df = pd.DataFrame(xf)
    xt_df = pd.DataFrame(xt)
    for i in cat_idx:
        xf_df[i] = xf_df[i].fillna(-1).astype(int).astype("category")
        xt_df[i] = pd.Categorical(xt_df[i].fillna(-1).astype(int), categories=xf_df[i].cat.categories)
    m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, random_state=1, n_jobs=-1)
    m.fit(xf_df, y)
    add("XGBoost", m.predict(xt_df), time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    cf = fit[num + cat].copy()
    ct = test[num + cat].copy()
    m = catboost.CatBoostRegressor(iterations=600, learning_rate=0.08, depth=8, loss_function="RMSE", random_seed=1,
                                   verbose=0, thread_count=-1)
    m.fit(cf, y, cat_features=cat)
    add("CatBoost", m.predict(ct), time.perf_counter() - t0, 1)

    tr, va, te, cards = nn_arrays(train, valid, test, num, cat)
    for s in (1, 2, 3, 4, 5):
        t0 = time.perf_counter()
        torch.manual_seed(s)
        model = TabMLP(len(num), cards)
        model, predict, hist = train_nn(model, tr, va, train["y_loss"].to_numpy(), valid["y_loss"].to_numpy(), seed=s)
        hist_rows += [{"model": "MLP", "seed": s, **h} for h in hist]
        add("MLP", predict(te, None), time.perf_counter() - t0, s)

    if not skip_gru:
        panel = load_panel()
        panel = panel.sort_values(["plant", "time"]).reset_index(drop=True)
        cols = ["loss", "rate", "coded", "gen", "ref", "avail"]
        vals = panel[cols].to_numpy(np.float32)
        vmu, vsd = np.nanmean(vals, 0), np.nanstd(vals, 0) + 1e-6
        vals = (vals - vmu) / vsd
        pos = pd.Series(np.arange(len(panel)), index=pd.MultiIndex.from_frame(panel[["plant", "time"]]))
        plant_start = panel.groupby("plant").cumcount().to_numpy()

        def seq_for(d):
            p = pos.reindex(pd.MultiIndex.from_frame(d[["plant", "time"]])).to_numpy().astype(int)
            s = sequences(None, vals, p, offset=-1, length=12)          # protocol B: window ends at t-1
            idx = p[:, None] - 1 + np.arange(-11, 1)[None, :]            # positions t-12 .. t-1
            first = (p - plant_start[p])[:, None]                        # first position of the plant
            s[idx < first] = 0
            return torch.tensor(s)
        seqs_tr, seqs_va, seqs_te = seq_for(train), seq_for(valid), seq_for(test)
        for s in (1, 2, 3):
            t0 = time.perf_counter()
            torch.manual_seed(s)
            model = SeqGRU(len(cols), len(num), cards)
            model, predict, hist = train_nn(model, tr, va, train["y_loss"].to_numpy(), valid["y_loss"].to_numpy(),
                                            seqs=(seqs_tr, seqs_va), seed=s, epochs=15)
            hist_rows += [{"model": "GRU", "seed": s, **h} for h in hist]
            add("GRU", predict(te, seqs_te), time.perf_counter() - t0, s)

    pd.DataFrame(regs).to_csv(OUT / "main_families_regression.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(OUT / "main_families_nn_history.csv", index=False)
    pd.DataFrame(rank_rows(test, scores, "families", ks=(10,), horizons=(1,))).to_csv(OUT / "main_families_ranking.csv", index=False)
    pd.Series(timing).to_csv(OUT / "main_timing.csv")
    keep = test[["time", "plant"]].copy()
    for n, s in scores.items():
        keep[n] = s
    keep.to_parquet(OUT / "main_families_scores.parquet", index=False)


def stage_policies(df, sets):
    train, valid, test = main_split(df)
    rows, regs = [], []
    for prot in (PROTOCOL_B, PROTOCOL_A):
        num, cat = feature_list(sets, prot)
        sc = Scorer(train, valid, test, num, cat, prot, log=log)
        scores = sc.baselines()
        scores.update(sc.hgb_scores())
        scores["FQI"] = sc.fitted_q()
        scores.update(sc.lightgbm_scores())
        rows += [{"protocol": prot.name, **r} for r in rank_rows(test, scores, "policies")]
        dd = pd.concat({n: daily_sums(test, s, 10, 1) for n, s in scores.items()}, names=["policy"])
        dd.to_csv(OUT / f"main_policies_daily_k10_{prot.name}.csv")
        dd6 = pd.concat({n: daily_sums(test, s, 10, 6) for n, s in scores.items()}, names=["policy"])
        dd6.to_csv(OUT / f"main_policies_daily_k10_h6_{prot.name}.csv")
        base = latest(test, "rate", prot).fillna(0).to_numpy()
        yt = test["y_rate"].to_numpy()
        regs.append({"protocol": prot.name, "target": "rate", "mae": mean_absolute_error(yt, sc.rate_pred),
                     "rmse": mean_squared_error(yt, sc.rate_pred) ** 0.5, "mae_pers": mean_absolute_error(yt, base),
                     "rmse_pers": mean_squared_error(yt, base) ** 0.5,
                     "r2": 1 - ((yt - sc.rate_pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum(),
                     "r2_pers": 1 - ((yt - base) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()})
        yl = test["y_loss"].to_numpy()
        bl = latest(test, "loss", prot).fillna(0).to_numpy()
        regs.append({"protocol": prot.name, "target": "loss_mwh", "mae": mean_absolute_error(yl, sc.loss_pred) * 0.5,
                     "rmse": mean_squared_error(yl, sc.loss_pred) ** 0.5 * 0.5, "mae_pers": mean_absolute_error(yl, bl) * 0.5,
                     "rmse_pers": mean_squared_error(yl, bl) ** 0.5 * 0.5,
                     "r2": 1 - ((yl - sc.loss_pred) ** 2).sum() / ((yl - yl.mean()) ** 2).sum(),
                     "r2_pers": 1 - ((yl - bl) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()})
        pd.Series(sc.times).to_csv(OUT / f"main_policies_fit_seconds_{prot.name}.csv")
        keep = test[["time", "plant", "state", "subsystem", "y_loss", "y_event", "y_reason", "loss", "loss_lag1"]
                    + [f"y_loss_h{h}" for h in range(1, 13)]].copy()
        for n, s in scores.items():
            keep["s_" + n] = s
        keep.to_parquet(OUT / f"main_policies_scores_{prot.name}.parquet", index=False)
        lg = getattr(sc, "lgbm_seed_preds", None)
    pd.DataFrame(rows).to_csv(OUT / "main_policies_ranking.csv", index=False)
    pd.DataFrame(regs).to_csv(OUT / "main_policies_forecast.csv", index=False)


def stage_ablation(df, sets):
    train, valid, test = main_split(df)
    fit = pd.concat([train, valid])
    rows = []
    variants = {"Protocol B (all groups)": (PROTOCOL_B, ()), "Protocol A (adds values at t)": (PROTOCOL_A, ()),
                "B without long lags": (PROTOCOL_B, ("lags",)), "B without plant/state/subsystem": (PROTOCOL_B, ("id",)),
                "B without restriction codes": (PROTOCOL_B, ("codes",)), "B without calendar": (PROTOCOL_B, ("calendar",)),
                "B without codes and plant ids": (PROTOCOL_B, ("codes", "id"))}
    for name, (prot, drop) in variants.items():
        num, cat = feature_list(sets, prot, drop)
        enc = Encoder(cat).fit(fit)
        m = hgb_regressor(num, cat).fit(_cat_fix(enc.transform(fit, num), len(num)), fit["y_loss"])
        p = np.clip(m.predict(_cat_fix(enc.transform(test, num), len(num))), 0, None)
        r = ranking_metrics(test, p, 10, 1)
        rows.append({"variant": name, "n_features": len(num) + len(cat), "mae_mwh": mean_absolute_error(test["y_loss"], p) * 0.5, **r})
        log(f"  {name}: eff {r['oracle_eff']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "main_ablation.csv", index=False)


def stage_horizon(df, sets):
    train, valid, test = main_split(df)
    fit = pd.concat([train, valid])
    num, cat = feature_list(sets, PROTOCOL_B)
    enc = Encoder(cat).fit(fit)
    xf, xt = _cat_fix(enc.transform(fit, num), len(num)), _cat_fix(enc.transform(test, num), len(num))
    rows = []
    for gamma in (0.5, 0.8, 0.92, 0.99):
        for H in (1, 3, 6, 12):
            if H == 1 and gamma != 0.92:
                continue
            m = hgb_regressor(num, cat).fit(xf, discounted_return(fit, gamma, H))
            p = m.predict(xt)
            for eval_h in sorted({1, H}):
                r = ranking_metrics(test, p, 10, eval_h)
                rows.append({"gamma": gamma, "train_horizon": H, "eval_horizon": eval_h, **r})
            log(f"  gamma {gamma} H {H}: eff h1 {rows[-1 if H == 1 else -2]['oracle_eff']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "main_horizon.csv", index=False)


def stage_fqi(df, sets):
    train, valid, test = main_split(df)
    num, cat = feature_list(sets, PROTOCOL_B)
    sc = Scorer(train, valid, test, num, cat, PROTOCOL_B, log=log)
    fit = sc.fit_df
    direct = hgb_regressor(num, cat).fit(sc.x_fit, discounted_return(fit)).predict(sc.x_test)
    rows = []
    for J in (1, 2, 3, 4, 6):
        q = sc.fitted_q(iterations=J)
        r = ranking_metrics(test, q, 10, 1)
        r6 = ranking_metrics(test, q, 10, 6)
        top_q = set(map(tuple, test[["time", "plant"]].to_numpy()[np.argsort(-q)][:0]))  # placeholder kept for clarity
        rows.append({"iterations": J, "spearman_vs_direct_return": spearmanr(q, direct).statistic,
                     "eff_h1": r["oracle_eff"], "eff_h6": r6["oracle_eff"], "ndcg_h1": r["ndcg_at_k"]})
        log(rows[-1])
    r = ranking_metrics(test, direct, 10, 1)
    r6 = ranking_metrics(test, direct, 10, 6)
    rows.append({"iterations": "direct return", "spearman_vs_direct_return": 1.0, "eff_h1": r["oracle_eff"],
                 "eff_h6": r6["oracle_eff"], "ndcg_h1": r["ndcg_at_k"]})
    pd.DataFrame(rows).to_csv(OUT / "main_fqi.csv", index=False)


def stage_coded(df, sets):
    train, valid, test = main_split(df)
    fit = pd.concat([train, valid])
    num, cat = feature_list(sets, PROTOCOL_B)
    enc = Encoder(cat).fit(fit)
    xf, xt = _cat_fix(enc.transform(fit, num), len(num)), _cat_fix(enc.transform(test, num), len(num))
    coded_target = fit["y_loss"] * fit["y_event"]
    m_all = hgb_regressor(num, cat).fit(xf, fit["y_loss"])
    m_cod = hgb_regressor(num, cat).fit(xf, coded_target)
    p_all, p_cod = np.clip(m_all.predict(xt), 0, None), np.clip(m_cod.predict(xt), 0, None)
    pers = test["loss_lag1"].fillna(0).to_numpy()
    rows = []
    t2 = test.copy()
    for label, gain in (("all loss", t2["y_loss"].to_numpy()), ("coded loss only", (t2["y_loss"] * t2["y_event"]).to_numpy()),
                        ("uncoded loss only", (t2["y_loss"] * (1 - t2["y_event"])).to_numpy())):
        tmp = t2.copy()
        tmp["y_loss"] = gain
        for name, s in (("HGB-loss (all records)", p_all), ("HGB-loss (coded target)", p_cod), ("Persistence", pers)):
            r = ranking_metrics(tmp, s, 10, 1)
            rows.append({"evaluation_gain": label, "policy": name, "share_of_test_loss": gain.sum() / t2["y_loss"].sum(), **r})
    # regression error by subset
    for label, mask in (("coded at t+1", t2["y_event"] == 1), ("uncoded at t+1", t2["y_event"] == 0)):
        rows.append({"evaluation_gain": "MAE subset " + label, "policy": "HGB-loss (all records)",
                     "mae_mwh": mean_absolute_error(t2.loc[mask, "y_loss"], p_all[mask.to_numpy()]) * 0.5,
                     "share_rows": mask.mean()})
    pd.DataFrame(rows).to_csv(OUT / "main_coded.csv", index=False)


STAGES = {"grid": stage_grid, "families": stage_families, "policies": stage_policies, "ablation": stage_ablation,
          "horizon": stage_horizon, "fqi": stage_fqi, "coded": stage_coded}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="grid,policies,ablation,horizon,fqi,coded,families")
    args = ap.parse_args()
    df, sets = get_features()
    for st in args.stages.split(","):
        log(f"### stage {st}")
        t0 = time.time()
        STAGES[st](df, sets)
        log(f"### stage {st} done in {time.time() - t0:.0f}s")

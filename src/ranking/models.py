"""Scoring models and ranking policies."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .common import GAMMA, HORIZON, Protocol, discounted_return, latest

HGB_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0)


class Encoder:
    """Integer codes for categorical columns; categories unseen in training map to -1."""

    def __init__(self, cat: list[str]):
        self.cat = cat
        self.maps: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame) -> "Encoder":
        for c in self.cat:
            self.maps[c] = {v: i for i, v in enumerate(sorted(df[c].unique()))}
        return self

    def transform(self, df: pd.DataFrame, num: list[str]) -> np.ndarray:
        parts = [df[num].to_numpy(dtype=np.float32)]
        for c in self.cat:
            parts.append(df[c].map(self.maps[c]).fillna(-1).to_numpy(dtype=np.float32)[:, None])
        return np.hstack(parts)


def hgb_regressor(num, cat, seed=42, **kw):
    params = {**HGB_PARAMS, **kw}
    mask = [False] * len(num) + [True] * len(cat)
    return HistGradientBoostingRegressor(categorical_features=mask, random_state=seed, **params)


def hgb_classifier(num, cat, seed=42, **kw):
    params = {**HGB_PARAMS, **kw}
    mask = [False] * len(num) + [True] * len(cat)
    return HistGradientBoostingClassifier(categorical_features=mask, random_state=seed, **params)


def _cat_fix(x: np.ndarray, n_num: int) -> np.ndarray:
    # HGB categorical features must be non-negative; unseen categories become missing
    x = x.copy()
    block = x[:, n_num:]
    block[block < 0] = np.nan
    x[:, n_num:] = block
    return x


class Scorer:
    """Fits all learned policies of one fold for one protocol and returns their test scores."""

    def __init__(self, train, valid, test, num, cat, protocol: Protocol, seed: int = 42, log=print):
        self.train, self.valid, self.test = train, valid, test
        self.num, self.cat, self.protocol, self.seed, self.log = num, cat, protocol, seed, log
        self.enc = Encoder(cat).fit(pd.concat([train, valid]))
        fit_df = pd.concat([train, valid])
        self.fit_df = fit_df
        self.x_fit = _cat_fix(self.enc.transform(fit_df, num), len(num))
        self.x_test = _cat_fix(self.enc.transform(test, num), len(num))
        self.times: dict[str, float] = {}

    def _timed(self, name, fn):
        t0 = time.perf_counter()
        out = fn()
        self.times[name] = time.perf_counter() - t0
        self.log(f"    {name}: {self.times[name]:.1f}s")
        return out

    # --- simple baselines -------------------------------------------------
    def baselines(self) -> dict[str, np.ndarray]:
        te, tr, p = self.test, self.fit_df, self.protocol
        rng = np.random.default_rng(self.seed)
        size = tr.groupby("plant")["ref"].max()
        hour_mean = tr.groupby(["plant", "hour"])["y_loss"].mean()
        recent = [latest(te, "loss", p)] + [te[f"loss_lag{h}"] for h in ([1, 2] if p.current else [2])]
        return {
            "Random": rng.random(len(te)),
            "Persistence": latest(te, "loss", p).fillna(0).to_numpy(),
            "Recent mean": pd.concat(recent, axis=1).mean(axis=1).fillna(0).to_numpy(),
            "Previous day": te["loss_lag47"].fillna(0).to_numpy(),
            "Plant-hour mean": te[["plant", "hour"]].merge(hour_mean.rename("m").reset_index(), on=["plant", "hour"],
                                                           how="left")["m"].fillna(0).to_numpy(),
            "Plant size": te["plant"].map(size).fillna(0).to_numpy(),
        }

    # --- learned scores -----------------------------------------------------
    def hgb_scores(self) -> dict[str, np.ndarray]:
        num, cat, fd, te, p = self.num, self.cat, self.fit_df, self.test, self.protocol
        scale = latest(te, "ref", p).fillna(0).to_numpy()
        out = {}

        def ev():
            m = hgb_classifier(num, cat, self.seed).fit(self.x_fit, fd["y_event"].astype(int))
            return m.predict_proba(self.x_test)[:, 1]
        out["HGB-event"] = self._timed("HGB-event", ev) * scale

        def rate():
            m = hgb_regressor(num, cat, self.seed).fit(self.x_fit, fd["y_rate"])
            return np.clip(m.predict(self.x_test), 0, 1)
        self.rate_pred = self._timed("HGB-rate", rate)
        out["HGB-rate"] = self.rate_pred * scale

        def loss():
            m = hgb_regressor(num, cat, self.seed).fit(self.x_fit, fd["y_loss"])
            return np.clip(m.predict(self.x_test), 0, None)
        self.loss_pred = self._timed("HGB-loss", loss)
        out["HGB-loss"] = self.loss_pred

        def ret():
            m = hgb_regressor(num, cat, self.seed).fit(self.x_fit, discounted_return(fd))
            return m.predict(self.x_test)
        out["HGB-return"] = self._timed("HGB-return", ret)
        return out

    def fitted_q(self, gamma: float = GAMMA, iterations: int = HORIZON) -> np.ndarray:
        """Iterative fitted-Q evaluation with bootstrapped Bellman targets.
        Selecting a plant does not change the next state, so Q(s,1) - Q(s,0) = E[L_{t+1} | s]
        and the iteration estimates the discounted future loss of the plant."""
        fd = self.fit_df.reset_index(drop=True)
        key = pd.MultiIndex.from_arrays([fd["plant"], fd["time"]])
        nxt = pd.MultiIndex.from_arrays([fd["plant"], fd["time"] + pd.Timedelta(minutes=30)])
        pos = pd.Series(np.arange(len(fd)), index=key)
        nxt_idx = pos.reindex(nxt).to_numpy()
        has_next = ~np.isnan(nxt_idx)
        nxt_idx = np.nan_to_num(nxt_idx).astype(int)
        r = fd["y_loss"].to_numpy()
        q_next = np.zeros(len(fd))
        model = None

        def run():
            nonlocal q_next, model
            for _ in range(iterations):
                y = r + gamma * np.where(has_next, q_next[nxt_idx], 0.0)
                model = hgb_regressor(self.num, self.cat, self.seed, max_iter=150).fit(self.x_fit, y)
                q_next = model.predict(self.x_fit)
            return model.predict(self.x_test)
        return self._timed("FQI", run)

    def lightgbm_scores(self, seeds=(42,)) -> dict[str, np.ndarray]:
        import lightgbm as lgb
        num, cat, fd = self.num, self.cat, self.fit_df
        cat_idx = list(range(len(num), len(num) + len(cat)))
        out = {}
        preds = []
        for s in seeds:
            def reg():
                m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                      random_state=s, verbose=-1)
                m.fit(self.x_fit, fd["y_loss"], categorical_feature=cat_idx)
                return np.clip(m.predict(self.x_test), 0, None)
            preds.append(self._timed(f"LGBM-loss s{s}", reg))
        out["LGBM-loss"] = preds[0]
        self.lgbm_seed_preds = preds

        # learning to rank: one query per instant, graded relevance from the next-interval loss
        order = np.argsort(fd["time"].to_numpy(), kind="stable")
        xs = self.x_fit[order]
        y = fd["y_loss"].to_numpy()[order]
        pos_losses = fd["y_loss"][fd["y_loss"] > 0.5]
        edges = np.quantile(pos_losses, np.linspace(0, 1, 21)[1:-1]) if len(pos_losses) else np.array([])
        label = np.where(y > 0.5, 1 + np.searchsorted(edges, y), 0).astype(int)
        group = fd["time"].to_numpy()[order]
        _, counts = np.unique(group, return_counts=True)

        def rank():
            m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=63,
                               min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               lambdarank_truncation_level=20, label_gain=list(range(0, 64)), random_state=self.seed,
                               verbose=-1)
            m.fit(xs, label, group=counts, categorical_feature=cat_idx)
            return m.predict(self.x_test)
        out["LambdaMART"] = self._timed("LambdaMART", rank)
        return out

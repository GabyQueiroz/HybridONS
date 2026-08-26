from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "restricao_coff_fotovoltaica_consolidado.parquet"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureSpec:
    categorical: list[str]
    numeric: list[str]

    @property
    def all(self) -> list[str]:
        return self.categorical + self.numeric


def prepare_dataset() -> tuple[pd.DataFrame, FeatureSpec]:
    df = pd.read_parquet(DATA)
    df = df.sort_values(["id_ons", "din_instante"]).reset_index(drop=True)
    df["din_instante"] = pd.to_datetime(df["din_instante"])
    df["mes"] = df["din_instante"].dt.month
    df["restricao_codificada"] = df["cod_razaorestricao"].astype(str).str.len() > 0
    df["perda_ref_mwmed"] = (df["val_geracaoreferencia"] - df["val_geracao"]).clip(lower=0)
    df["taxa_ref"] = np.where(
        df["val_geracaoreferencia"] > 0,
        df["perda_ref_mwmed"] / df["val_geracaoreferencia"],
        np.nan,
    )

    for lag in [1, 2, 6, 12, 48, 96]:
        group = df.groupby("id_ons", sort=False)
        df[f"taxa_ref_lag_{lag}"] = group["taxa_ref"].shift(lag)
        df[f"perda_ref_lag_{lag}"] = group["perda_ref_mwmed"].shift(lag)
        df[f"restricao_lag_{lag}"] = group["restricao_codificada"].shift(lag)
        df[f"geracao_lag_{lag}"] = group["val_geracao"].shift(lag)

    # Alvos futuros: 1 intervalo a frente para previsao; 6 intervalos para decisao.
    group = df.groupby("id_ons", sort=False)
    df["target_taxa_h1"] = group["taxa_ref"].shift(-1)
    df["target_evento_h1"] = group["restricao_codificada"].shift(-1)
    df["target_perda_h1"] = group["perda_ref_mwmed"].shift(-1)
    future_losses = [group["perda_ref_mwmed"].shift(-h).fillna(0) for h in range(1, 7)]
    df["target_retorno_h6"] = sum((0.92 ** (h - 1)) * future_losses[h - 1] for h in range(1, 7))

    model_df = df[
        (df["val_geracaoreferencia"] > 0)
        & df["target_taxa_h1"].notna()
        & df["taxa_ref_lag_96"].notna()
    ].copy()

    categorical = ["id_ons", "id_estado", "id_subsistema", "cod_razaorestricao", "cod_origemrestricao"]
    numeric = [
        "hora",
        "minuto",
        "dia_semana",
        "mes",
        "val_geracao",
        "val_geracaolimitada",
        "val_disponibilidade",
        "val_geracaoreferencia",
        "taxa_ref_lag_1",
        "taxa_ref_lag_2",
        "taxa_ref_lag_6",
        "taxa_ref_lag_12",
        "taxa_ref_lag_48",
        "taxa_ref_lag_96",
        "perda_ref_lag_1",
        "perda_ref_lag_2",
        "perda_ref_lag_6",
        "perda_ref_lag_12",
        "perda_ref_lag_48",
        "perda_ref_lag_96",
        "restricao_lag_1",
        "restricao_lag_2",
        "restricao_lag_6",
        "restricao_lag_12",
        "restricao_lag_48",
        "restricao_lag_96",
        "geracao_lag_1",
        "geracao_lag_48",
    ]

    for col in categorical:
        model_df[col] = model_df[col].astype(str).fillna("")
    for col in numeric:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

    model_df["target_evento_h1"] = model_df["target_evento_h1"].astype(bool)
    return model_df, FeatureSpec(categorical, numeric)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["din_instante"] < "2026-04-01"].copy()
    valid = df[(df["din_instante"] >= "2026-04-01") & (df["din_instante"] < "2026-07-01")].copy()
    test = df[df["din_instante"] >= "2026-07-01"].copy()
    if len(train) > 280_000:
        train = train.sample(n=280_000, random_state=SEED).sort_values("din_instante")
    return train, valid, test


def preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), spec.categorical),
            ("num", "passthrough", spec.numeric),
        ]
    )


def evaluate_regression(name: str, y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict:
    pred = np.clip(pred, 0, 1)
    baseline = np.clip(baseline, 0, 1)
    return {
        "modelo": name,
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "mae_persistencia": float(mean_absolute_error(y, baseline)),
        "rmse_persistencia": float(mean_squared_error(y, baseline) ** 0.5),
        "r2_persistencia": float(r2_score(y, baseline)),
    }


def evaluate_classifier(name: str, y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba >= 0.5
    p, r, _, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "modelo": name,
        "n": int(len(y)),
        "prevalencia": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "precision_05": float(p),
        "recall_05": float(r),
        "f1_05": float(f1_score(y, pred)),
    }


def train_gradient_boosting(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, dict, dict]:
    grid = [
        {"max_iter": 70, "learning_rate": 0.06, "max_leaf_nodes": 31},
        {"max_iter": 110, "learning_rate": 0.05, "max_leaf_nodes": 47},
    ]
    rows = []
    best_reg = None
    best_clf = None
    best_reg_score = math.inf
    best_clf_score = -math.inf
    for hp in grid:
        print(f"[GB] Treinando hiperparametros: {hp}", flush=True)
        reg = Pipeline(
            [
                ("pre", preprocessor(spec)),
                ("model", HistGradientBoostingRegressor(random_state=SEED, **hp)),
            ]
        )
        clf = Pipeline(
            [
                ("pre", preprocessor(spec)),
                ("model", HistGradientBoostingClassifier(random_state=SEED, **hp)),
            ]
        )
        reg.fit(train[spec.all], train["target_taxa_h1"])
        clf.fit(train[spec.all], train["target_evento_h1"])
        valid_reg = np.clip(reg.predict(valid[spec.all]), 0, 1)
        valid_clf = clf.predict_proba(valid[spec.all])[:, 1]
        mae = mean_absolute_error(valid["target_taxa_h1"], valid_reg)
        ap = average_precision_score(valid["target_evento_h1"].astype(int), valid_clf)
        rows.append({"familia": "HistGradientBoosting", **hp, "valid_mae_taxa": mae, "valid_average_precision": ap})
        if mae < best_reg_score:
            best_reg_score = mae
            best_reg = reg
        if ap > best_clf_score:
            best_clf_score = ap
            best_clf = clf

    assert best_reg is not None and best_clf is not None
    metrics = {
        "valid_reg": evaluate_regression(
            "HistGradientBoosting",
            valid["target_taxa_h1"].to_numpy(),
            best_reg.predict(valid[spec.all]),
            valid["taxa_ref_lag_1"].to_numpy(),
        ),
        "test_reg": evaluate_regression(
            "HistGradientBoosting",
            test["target_taxa_h1"].to_numpy(),
            best_reg.predict(test[spec.all]),
            test["taxa_ref_lag_1"].to_numpy(),
        ),
        "valid_clf": evaluate_classifier(
            "HistGradientBoosting",
            valid["target_evento_h1"].astype(int).to_numpy(),
            best_clf.predict_proba(valid[spec.all])[:, 1],
        ),
        "test_clf": evaluate_classifier(
            "HistGradientBoosting",
            test["target_evento_h1"].astype(int).to_numpy(),
            best_clf.predict_proba(test[spec.all])[:, 1],
        ),
    }
    models = {"reg": best_reg, "clf": best_clf}
    return pd.DataFrame(rows), metrics, models


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def encode_for_mlp(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, spec: FeatureSpec):
    prep = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), spec.categorical),
            ("num", StandardScaler(), spec.numeric),
        ]
    )
    x_train = prep.fit_transform(train[spec.all]).astype("float32")
    x_valid = prep.transform(valid[spec.all]).astype("float32")
    x_test = prep.transform(test[spec.all]).astype("float32")
    return prep, x_train, x_valid, x_test


def train_mlp(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, dict, MLP]:
    prep, x_train, x_valid, x_test = encode_for_mlp(train, valid, test, spec)
    # Subamostra para busca neural em CPU; validacao/teste permanecem temporais completos.
    max_n = min(len(x_train), 80_000)
    idx = np.random.default_rng(SEED).choice(len(x_train), max_n, replace=False)
    xtr = torch.tensor(x_train[idx])
    y_reg = torch.tensor(train["target_taxa_h1"].to_numpy()[idx].astype("float32")).reshape(-1, 1)
    y_cls = torch.tensor(train["target_evento_h1"].to_numpy()[idx].astype("float32")).reshape(-1, 1)
    xv = torch.tensor(x_valid)
    yv_reg = valid["target_taxa_h1"].to_numpy()
    yv_cls = valid["target_evento_h1"].astype(int).to_numpy()

    configs = [
        {"hidden": 64, "dropout": 0.05, "lr": 1e-3, "epochs": 5},
        {"hidden": 128, "dropout": 0.10, "lr": 8e-4, "epochs": 8},
        {"hidden": 192, "dropout": 0.15, "lr": 6e-4, "epochs": 10},
    ]
    rows = []
    best = None
    best_score = math.inf
    best_history = []
    for cfg in configs:
        print(f"[MLP] Treinando configuracao: {cfg}", flush=True)
        model = MLP(xtr.shape[1], cfg["hidden"], cfg["dropout"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        batch = 4096
        history = []
        for epoch in range(1, cfg["epochs"] + 1):
            perm = torch.randperm(len(xtr))
            model.train()
            losses = []
            for start in range(0, len(xtr), batch):
                b = perm[start : start + batch]
                out = model(xtr[b])
                loss_reg = torch.nn.functional.mse_loss(torch.sigmoid(out[:, :1]), y_reg[b])
                loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 1:2], y_cls[b])
                loss = loss_reg + 0.35 * loss_cls
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                pred_v = torch.sigmoid(model(xv))
                mae = mean_absolute_error(yv_reg, pred_v[:, 0].numpy())
                ap = average_precision_score(yv_cls, pred_v[:, 1].numpy())
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_mae_taxa": mae, "valid_average_precision": ap, **cfg})
            print(f"[MLP] epoch={epoch} loss={np.mean(losses):.5f} val_mae={mae:.5f} val_ap={ap:.5f}", flush=True)
        rows.extend(history)
        if history[-1]["valid_mae_taxa"] < best_score:
            best_score = history[-1]["valid_mae_taxa"]
            best = model
            best_history = history

    assert best is not None
    best.eval()
    with torch.no_grad():
        pv = torch.sigmoid(best(torch.tensor(x_valid))).numpy()
        pt = torch.sigmoid(best(torch.tensor(x_test))).numpy()
    metrics = {
        "valid_reg": evaluate_regression("MLP", yv_reg, pv[:, 0], valid["taxa_ref_lag_1"].to_numpy()),
        "test_reg": evaluate_regression("MLP", test["target_taxa_h1"].to_numpy(), pt[:, 0], test["taxa_ref_lag_1"].to_numpy()),
        "valid_clf": evaluate_classifier("MLP", yv_cls, pv[:, 1]),
        "test_clf": evaluate_classifier("MLP", test["target_evento_h1"].astype(int).to_numpy(), pt[:, 1]),
        "best_history": best_history,
    }
    # Guardar preprocessor no modelo para previsoes posteriores neste processo.
    best.preprocessor = prep  # type: ignore[attr-defined]
    return pd.DataFrame(rows), metrics, best


def decision_policy_table(test: pd.DataFrame, gb_models: dict, mlp: MLP, spec: FeatureSpec) -> pd.DataFrame:
    out = test[["din_instante", "ano_mes", "id_ons", "id_estado", "id_subsistema", "target_perda_h1", "target_retorno_h6", "perda_ref_lag_1", "perda_ref_lag_48"]].copy()
    out["score_persistencia"] = out["perda_ref_lag_1"].fillna(0)
    out["score_sazonal_diaria"] = out["perda_ref_lag_48"].fillna(0)
    out["score_gb_taxa"] = np.clip(gb_models["reg"].predict(test[spec.all]), 0, 1) * test["val_geracaoreferencia"].to_numpy()
    out["score_gb_evento"] = gb_models["clf"].predict_proba(test[spec.all])[:, 1] * test["val_geracaoreferencia"].to_numpy()
    with torch.no_grad():
        x = mlp.preprocessor.transform(test[spec.all]).astype("float32")  # type: ignore[attr-defined]
        pred = torch.sigmoid(mlp(torch.tensor(x))).numpy()
    out["score_mlp_hibrido"] = pred[:, 0] * pred[:, 1] * test["val_geracaoreferencia"].to_numpy()

    train_proxy = test.copy()
    train_proxy["q_target"] = train_proxy["target_retorno_h6"]
    return out


def train_offline_q(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    q_train = train.copy()
    if len(q_train) > 50_000:
        q_train = q_train.sample(n=50_000, random_state=SEED)
    prep = preprocessor(spec)
    xq = prep.fit_transform(q_train[spec.all])
    q_model = HistGradientBoostingRegressor(
        max_iter=35,
        learning_rate=0.06,
        max_leaf_nodes=31,
        random_state=SEED,
    )
    q_model.fit(xq, q_train["target_retorno_h6"])
    return q_model.predict(prep.transform(test[spec.all]))


def evaluate_topk(decisions: pd.DataFrame, ks=(5, 10, 20)) -> pd.DataFrame:
    score_cols = [c for c in decisions.columns if c.startswith("score_")]
    rows = []
    total_loss = float(decisions["target_perda_h1"].sum())
    for k in ks:
        for policy in score_cols + ["random"]:
            tmp = decisions[["din_instante", "target_perda_h1"]].copy()
            if policy == "random":
                rng = np.random.default_rng(SEED + k)
                tmp["_score"] = rng.random(len(tmp))
            else:
                tmp["_score"] = decisions[policy].to_numpy()
            tmp["_rank_policy"] = tmp.groupby("din_instante")["_score"].rank(method="first", ascending=False)
            tmp["_rank_oracle"] = tmp.groupby("din_instante")["target_perda_h1"].rank(method="first", ascending=False)
            cap = float(tmp.loc[tmp["_rank_policy"] <= k, "target_perda_h1"].sum())
            ora = float(tmp.loc[tmp["_rank_oracle"] <= k, "target_perda_h1"].sum())
            tot = total_loss
            rows.append(
                {
                    "k": k,
                    "politica": policy.replace("score_", ""),
                    "perda_capturada": cap,
                    "perda_oraculo_topk": ora,
                    "perda_total": tot,
                    "captura_vs_total": cap / tot if tot else 0,
                    "eficiencia_vs_oraculo": cap / ora if ora else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["k", "eficiencia_vs_oraculo"], ascending=[True, False])


def make_plots(model_rows: pd.DataFrame, mlp_history: pd.DataFrame, policy: pd.DataFrame, monthly: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    best_cfg = mlp_history.sort_values(["valid_mae_taxa", "epoch"]).iloc[0][["hidden", "dropout", "lr", "epochs"]].to_dict()
    subset = mlp_history[
        (mlp_history["hidden"] == best_cfg["hidden"])
        & (mlp_history["dropout"] == best_cfg["dropout"])
        & (mlp_history["lr"] == best_cfg["lr"])
    ]
    sns.lineplot(data=subset, x="epoch", y="train_loss", marker="o", label="Treino")
    ax2 = plt.gca().twinx()
    sns.lineplot(data=subset, x="epoch", y="valid_mae_taxa", marker="s", color="#b23a48", ax=ax2, label="Validacao MAE")
    plt.title("Curva de aprendizado do MLP hibrido")
    plt.tight_layout()
    plt.savefig(FIGS / "curva_aprendizado_mlp.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    p10 = policy[policy["k"] == 10].sort_values("eficiencia_vs_oraculo", ascending=False)
    sns.barplot(data=p10, y="politica", x="eficiencia_vs_oraculo", color="#2b6c7e")
    plt.xlabel("Eficiencia em relacao ao oraculo top-10")
    plt.ylabel("Politica")
    plt.title("Comparacao de politicas de priorizacao no teste")
    plt.tight_layout()
    plt.savefig(FIGS / "politicas_top10.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly, x="ano_mes", y="captura_vs_total", hue="politica", marker="o")
    plt.xticks(rotation=30)
    plt.xlabel("Mes de teste")
    plt.ylabel("Captura da perda total")
    plt.title("Robustez temporal das politicas")
    plt.tight_layout()
    plt.savefig(FIGS / "robustez_mensal_politicas.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=model_rows, x="valid_mae_taxa", y="valid_average_precision", hue="familia", s=90)
    plt.xlabel("MAE de validacao")
    plt.ylabel("Average precision de validacao")
    plt.title("Busca de hiperparametros")
    plt.tight_layout()
    plt.savefig(FIGS / "busca_hiperparametros.png", dpi=220)
    plt.close()


def monthly_policy(decisions: pd.DataFrame, best_policies: list[str], k: int = 10) -> pd.DataFrame:
    rows = []
    for month, month_df in decisions.groupby("ano_mes"):
        eval_df = evaluate_topk(month_df, ks=(k,))
        eval_df = eval_df[eval_df["politica"].isin(best_policies)]
        eval_df["ano_mes"] = month
        rows.append(eval_df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print("[1/6] Preparando dados e features temporais", flush=True)
    df, spec = prepare_dataset()
    train, valid, test = temporal_split(df)
    split_info = {
        "train_inicio": str(train["din_instante"].min()),
        "train_fim": str(train["din_instante"].max()),
        "valid_inicio": str(valid["din_instante"].min()),
        "valid_fim": str(valid["din_instante"].max()),
        "test_inicio": str(test["din_instante"].min()),
        "test_fim": str(test["din_instante"].max()),
        "train_n": int(len(train)),
        "valid_n": int(len(valid)),
        "test_n": int(len(test)),
    }

    print("[2/6] Treinando modelos Gradient Boosting", flush=True)
    gb_grid, gb_metrics, gb_models = train_gradient_boosting(train, valid, test, spec)
    print("[3/6] Treinando MLP neural", flush=True)
    mlp_history, mlp_metrics, mlp_model = train_mlp(train, valid, test, spec)

    print("[4/6] Simulando politicas de decisao", flush=True)
    decisions = decision_policy_table(test, gb_models, mlp_model, spec)
    print("[5/6] Treinando score offline fitted-Q", flush=True)
    decisions["score_offline_q"] = train_offline_q(train, valid, test, spec)
    print("[5/6] Avaliando politicas top-k", flush=True)
    policy_metrics = evaluate_topk(decisions, ks=(5, 10, 20))
    best_policies = policy_metrics[policy_metrics["k"] == 10].head(4)["politica"].tolist()
    monthly = monthly_policy(decisions, best_policies, k=10)

    gb_grid.to_csv(TABLES / "grid_gradient_boosting.csv", index=False)
    mlp_history.to_csv(TABLES / "grid_mlp_epocas.csv", index=False)
    policy_metrics.to_csv(TABLES / "policy_topk_metrics.csv", index=False)
    monthly.to_csv(TABLES / "policy_monthly_robustness.csv", index=False)
    decisions.sample(n=min(5000, len(decisions)), random_state=SEED).to_csv(TABLES / "exemplos_decisao_teste.csv", index=False)

    all_metrics = {
        "split_temporal": split_info,
        "gradient_boosting": gb_metrics,
        "mlp": {k: v for k, v in mlp_metrics.items() if k != "best_history"},
        "melhores_politicas_top10": policy_metrics[policy_metrics["k"] == 10].head(6).to_dict(orient="records"),
    }
    with open(TABLES / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    model_rows = pd.concat([gb_grid, mlp_history.groupby(["hidden", "dropout", "lr", "epochs"], as_index=False).tail(1).assign(familia="MLP")], ignore_index=True, sort=False)
    print("[6/6] Gerando tabelas e figuras", flush=True)
    make_plots(model_rows, mlp_history, policy_metrics, monthly)
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

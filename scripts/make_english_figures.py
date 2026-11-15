from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT_REPO = ROOT / "results" / "figures_en"
OUT_DOWNLOADS = Path.home() / "Downloads"

OUT_REPO.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

POLICY_LABELS = {
    "gb_taxa": "GB-rate",
    "offline_q": "Offline-Q",
    "mlp_hibrido": "Hybrid MLP",
    "persistencia": "Persistence",
    "gb_evento": "GB-event",
    "sazonal_diaria": "Daily seasonal",
    "random": "Random",
}


def save_figure(name: str) -> None:
    for out_dir in (OUT_REPO, OUT_DOWNLOADS):
        plt.savefig(out_dir / name, dpi=220, bbox_inches="tight")
    plt.close()


def monthly_policy_stability() -> None:
    df = pd.read_csv(TABLES / "policy_monthly_robustness.csv")
    order = ["gb_taxa", "offline_q", "mlp_hibrido", "persistencia"]
    df = df[(df["k"] == 10) & (df["politica"].isin(order))].copy()
    df["Policy"] = df["politica"].map(POLICY_LABELS)
    df["Test month"] = df["ano_mes"]

    plt.figure(figsize=(11, 5.5))
    ax = sns.lineplot(
        data=df,
        x="Test month",
        y="captura_vs_total",
        hue="Policy",
        hue_order=[POLICY_LABELS[p] for p in order],
        marker="o",
        linewidth=2,
    )
    ax.set_title("Temporal stability of prioritization policies")
    ax.set_xlabel("Test month")
    ax.set_ylabel("Share of total loss captured")
    ax.legend(title="Policy", loc="lower center", frameon=True)
    save_figure("robustez_mensal_politicas_en.png")


def top10_policy_comparison() -> None:
    df = pd.read_csv(TABLES / "policy_topk_metrics.csv")
    df = df[df["k"] == 10].copy()
    df["Policy"] = df["politica"].map(POLICY_LABELS)
    df = df.sort_values("eficiencia_vs_oraculo", ascending=True)

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=df,
        x="eficiencia_vs_oraculo",
        y="Policy",
        color="#386c78",
    )
    ax.set_title("Comparison of prioritization policies on the test set")
    ax.set_xlabel("Efficiency relative to the top-10 oracle")
    ax.set_ylabel("Policy")
    ax.set_xlim(0, 1.0)
    save_figure("politicas_top10_en.png")


def top_plants() -> None:
    df = pd.read_csv(TABLES / "eda_top_usinas.csv").head(15).copy()
    df = df.sort_values("perda_ref_mwmed", ascending=True)

    plt.figure(figsize=(11, 6.6))
    ax = sns.barplot(
        data=df,
        x="perda_ref_mwmed",
        y="label",
        color="#55708d",
    )
    ax.set_title("Top 15 plants/groups by derived loss")
    ax.set_xlabel("Derived loss from reference generation")
    ax.set_ylabel("Plant/group")
    save_figure("eda_top_usinas_en.png")


def intraday_profile() -> None:
    df = pd.read_csv(TABLES / "hora.csv").copy()

    plt.figure(figsize=(11, 5.5))
    ax = sns.lineplot(
        data=df,
        x="hora",
        y="perda_ref_final_mwmed",
        marker="o",
        color="#735484",
        linewidth=2,
    )
    ax.set_title("Intraday loss profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Derived loss from reference generation")
    ax.set_xlim(-0.5, 23.5)
    save_figure("eda_perfil_intradiario_en.png")


def loss_by_reason() -> None:
    df = pd.read_csv(TABLES / "eda_razao.csv").copy()
    df["Restriction reason"] = df["cod_razaorestricao"].fillna("No code")
    df["Restriction reason"] = df["Restriction reason"].replace({"Sem codigo": "No code", "Sem código": "No code"})

    plt.figure(figsize=(8.8, 5.5))
    ax = sns.barplot(
        data=df,
        x="Restriction reason",
        y="perda_ref_mwmed",
        color="#916856",
    )
    ax.set_title("Loss by restriction reason")
    ax.set_xlabel("Restriction reason")
    ax.set_ylabel("Derived loss from reference generation")
    save_figure("eda_perda_razao_en.png")


def monthly_loss() -> None:
    df = pd.read_csv(TABLES / "eda_mensal.csv").copy()

    plt.figure(figsize=(13.2, 5.5))
    ax = sns.lineplot(
        data=df,
        x="ano_mes",
        y="perda_ref_mwmed",
        marker="o",
        color="#255e73",
        linewidth=2,
    )
    ax.set_title("Monthly photovoltaic loss due to constrained-off")
    ax.set_xlabel("Month")
    ax.set_ylabel("Derived loss from reference generation")
    ax.tick_params(axis="x", rotation=45)
    save_figure("eda_perda_mensal_en.png")


def loss_by_state() -> None:
    df = pd.read_csv(TABLES / "eda_estado.csv").copy()
    df = df.sort_values("perda_ref_mwmed", ascending=True)

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=df,
        x="perda_ref_mwmed",
        y="id_estado",
        color="#62764d",
    )
    ax.set_title("Spatial distribution of loss")
    ax.set_xlabel("Derived loss from reference generation")
    ax.set_ylabel("State")
    save_figure("eda_perda_estado_en.png")


def mlp_learning_curve() -> None:
    df = pd.read_csv(TABLES / "grid_mlp_epocas.csv").copy()
    best = df.sort_values(["epochs", "hidden"]).iloc[-1]
    df = df[
        (df["hidden"] == best["hidden"])
        & (df["dropout"] == best["dropout"])
        & (df["lr"] == best["lr"])
        & (df["epochs"] == best["epochs"])
    ].copy()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        df["epoch"],
        df["train_loss"],
        marker="o",
        color="#4C72B0",
        linewidth=2,
        label="Training loss",
    )
    line2 = ax2.plot(
        df["epoch"],
        df["valid_mae_taxa"],
        marker="s",
        color="#B53A4A",
        linewidth=2,
        label="Validation MAE",
    )

    ax1.set_title("Hybrid MLP learning curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax2.set_ylabel("Validation MAE")
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)
    fig.tight_layout()
    save_figure("curva_aprendizado_mlp_en.png")


def hyperparameter_search() -> None:
    gb = pd.read_csv(TABLES / "grid_gradient_boosting.csv").copy()
    mlp = pd.read_csv(TABLES / "grid_mlp_epocas.csv").copy()
    mlp = (
        mlp.sort_values("epoch")
        .groupby(["hidden", "dropout", "lr", "epochs"], as_index=False)
        .tail(1)
    )
    gb["Family"] = "HistGradientBoosting"
    mlp["Family"] = "MLP"
    df = pd.concat(
        [
            gb[["valid_mae_taxa", "valid_average_precision", "Family"]],
            mlp[["valid_mae_taxa", "valid_average_precision", "Family"]],
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(8.8, 5.5))
    ax = sns.scatterplot(
        data=df,
        x="valid_mae_taxa",
        y="valid_average_precision",
        hue="Family",
        s=110,
    )
    ax.set_title("Hyperparameter search")
    ax.set_xlabel("Validation MAE")
    ax.set_ylabel("Validation average precision")
    ax.legend(title="Family", loc="upper right", frameon=True)
    save_figure("busca_hiperparametros_en.png")


def main() -> None:
    monthly_policy_stability()
    top10_policy_comparison()
    top_plants()
    intraday_profile()
    loss_by_reason()
    monthly_loss()
    loss_by_state()
    mlp_learning_curve()
    hyperparameter_search()


if __name__ == "__main__":
    main()

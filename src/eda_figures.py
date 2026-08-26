from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "restricao_coff_fotovoltaica_consolidado.parquet"
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_parquet(DATA)
    df["din_instante"] = pd.to_datetime(df["din_instante"])
    df["ano_mes"] = df["din_instante"].dt.to_period("M").astype(str)
    df["perda_ref_mwmed"] = (df["val_geracaoreferencia"] - df["val_geracao"]).clip(lower=0)
    df["taxa_ref"] = df["perda_ref_mwmed"] / df["val_geracaoreferencia"].where(df["val_geracaoreferencia"] > 0)

    mensal = df.groupby("ano_mes", as_index=False).agg(
        perda_ref_mwmed=("perda_ref_mwmed", "sum"),
        taxa_media=("taxa_ref", "mean"),
        usinas=("id_ons", "nunique"),
    )
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=mensal, x="ano_mes", y="perda_ref_mwmed", marker="o", color="#24566b")
    plt.xticks(rotation=40)
    plt.xlabel("Mes")
    plt.ylabel("Perda derivada pela referencia")
    plt.title("Evolucao mensal da perda fotovoltaica por constrained-off")
    plt.tight_layout()
    plt.savefig(FIGS / "eda_perda_mensal.png", dpi=220)
    plt.close()

    estado = df.groupby("id_estado", as_index=False).agg(
        perda_ref_mwmed=("perda_ref_mwmed", "sum"),
        taxa_media=("taxa_ref", "mean"),
        usinas=("id_ons", "nunique"),
    ).sort_values("perda_ref_mwmed", ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=estado, x="perda_ref_mwmed", y="id_estado", color="#607744")
    plt.xlabel("Perda derivada pela referencia")
    plt.ylabel("Estado")
    plt.title("Distribuicao espacial da perda")
    plt.tight_layout()
    plt.savefig(FIGS / "eda_perda_estado.png", dpi=220)
    plt.close()

    razao = df.groupby("cod_razaorestricao", as_index=False).agg(
        perda_ref_mwmed=("perda_ref_mwmed", "sum"),
        linhas=("id_ons", "size"),
    )
    razao["cod_razaorestricao"] = razao["cod_razaorestricao"].replace("", "Sem codigo")
    razao = razao.sort_values("perda_ref_mwmed", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=razao, x="cod_razaorestricao", y="perda_ref_mwmed", color="#8f5d46")
    plt.xlabel("Razao da restricao")
    plt.ylabel("Perda derivada pela referencia")
    plt.title("Perda por razao de restricao")
    plt.tight_layout()
    plt.savefig(FIGS / "eda_perda_razao.png", dpi=220)
    plt.close()

    hora = df.groupby(["hora", "minuto"], as_index=False).agg(perda_ref_mwmed=("perda_ref_mwmed", "sum"))
    hora["slot"] = hora["hora"] + hora["minuto"] / 60
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hora, x="slot", y="perda_ref_mwmed", marker="o", color="#6f4e7c")
    plt.xlabel("Hora do dia")
    plt.ylabel("Perda derivada pela referencia")
    plt.title("Perfil intradiario da perda")
    plt.tight_layout()
    plt.savefig(FIGS / "eda_perfil_intradiario.png", dpi=220)
    plt.close()

    top = df.groupby(["id_ons", "nom_usina", "id_estado"], as_index=False).agg(
        perda_ref_mwmed=("perda_ref_mwmed", "sum"),
        taxa_media=("taxa_ref", "mean"),
    ).sort_values("perda_ref_mwmed", ascending=False).head(15)
    top["label"] = top["id_ons"] + " - " + top["id_estado"]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="perda_ref_mwmed", y="label", color="#466a8f")
    plt.xlabel("Perda derivada pela referencia")
    plt.ylabel("Usina/conjunto")
    plt.title("Top 15 usinas/conjuntos por perda")
    plt.tight_layout()
    plt.savefig(FIGS / "eda_top_usinas.png", dpi=220)
    plt.close()

    mensal.to_csv(TABLES / "eda_mensal.csv", index=False)
    estado.to_csv(TABLES / "eda_estado.csv", index=False)
    razao.to_csv(TABLES / "eda_razao.csv", index=False)
    top.to_csv(TABLES / "eda_top_usinas.csv", index=False)


if __name__ == "__main__":
    main()

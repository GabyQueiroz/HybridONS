# HybridONS

Projeto reprodutivel para analise de restricoes de operacao por constrained-off em usinas fotovoltaicas do SIN usando dados abertos da ONS.

## Titulo sugerido do artigo

**Priorizacao Operacional de Restricoes Constrained-Off em Usinas Fotovoltaicas por Modelo Hibrido de Previsao Temporal e Reinforcement Learning Offline**

Versao em ingles:

**Hybrid Temporal Forecasting and Offline Reinforcement Learning for Operational Prioritization of Photovoltaic Constrained-Off Events in the Brazilian Power System**

## Fonte dos dados

Dataset ONS: <https://dados.ons.org.br/dataset/restricao_coff_fotovoltaica>

Foram usados apenas os arquivos da propria ONS: Parquet mensal, dicionario JSON/PDF e metadados CKAN. O consolidado local cobre 2024-04-01 a 2026-08-25, com 2.830.896 registros semi-horarios.

## Como executar

```bash
pip install -r requirements.txt
python scripts/run_experiments.py
```

Os resultados sao salvos em:

- `results/tables/`: metricas, tabelas de apoio, exemplos de decisoes e resumo JSON.
- `results/figures/`: graficos prontos para uso no artigo.
- `data/processed/restricao_coff_fotovoltaica_consolidado.parquet`: base consolidada.

## Formulacao metodologica

O dataset nao contem acoes de controle, topologia de rede, carga, preco, clima ou despacho. Por isso, a abordagem mais defensavel nao e alegar controle fisico direto do SIN, mas sim suporte decisorio sequencial:

- prever risco e intensidade de constrained-off por usina/conjunto;
- selecionar top-k usinas para monitoramento/priorizacao operacional;
- avaliar quanto da perda futura observada e capturada pela politica.

## Modelos implementados

- Baseline de persistencia.
- HistGradientBoosting com busca de hiperparametros.
- MLP neural em PyTorch com diferentes epocas, tamanhos de camada, dropout e taxa de aprendizado.
- Politicas de priorizacao top-k:
  - persistencia;
  - sazonalidade diaria;
  - previsao por Gradient Boosting;
  - previsao hibrida MLP;
  - offline fitted-Q;
  - random;
  - oraculo usado apenas como teto superior de avaliacao.

## Principais metricas

- Regressao: MAE, RMSE, R2 para taxa de restricao no proximo intervalo.
- Classificacao: ROC-AUC, average precision, precision/recall/F1.
- Decisao: perda capturada, captura versus perda total e eficiencia versus oraculo top-k.

## Figuras

- `curva_aprendizado_mlp.png`: curva de aprendizado do MLP.
- `busca_hiperparametros.png`: comparacao de hiperparametros.
- `politicas_top10.png`: desempenho das politicas top-10.
- `robustez_mensal_politicas.png`: robustez temporal no periodo de teste.

## Uso exemplo

Cada linha em `results/tables/exemplos_decisao_teste.csv` representa uma decisao candidata por usina e instante. Para uso operacional, ordene as usinas de cada `din_instante` por `score_offline_q`, `score_gb_taxa` ou `score_mlp_hibrido` e escolha as top-k conforme o orcamento de monitoramento.

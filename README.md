# HybridONS

Reproducible project for the analysis of constrained-off operational restrictions in photovoltaic power plants of the Brazilian Interconnected Power System (SIN) using open data from ONS.

## Article

**Operational Prioritization of Constrained-Off Restrictions in Photovoltaic Power Plants through a Hybrid Temporal Forecasting and Offline Reinforcement Learning Model**

## Data source

ONS dataset: <https://dados.ons.org.br/dataset/restricao_coff_fotovoltaica>

Files from ONS itself were used: monthly Parquet files, JSON/PDF dictionary, and CKAN metadata. The local consolidated dataset covers 2024-04-01 to 2026-08-25, with 2,830,896 half-hourly records.

## How to run

```bash
pip install -r requirements.txt
python scripts/run_experiments.py

## Implemented models - Persistence baseline. - HistGradientBoosting with hyperparameter search. - PyTorch MLP with varying epochs, layer sizes, dropout, and learning rates. - Top-k prioritization policies: - persistence; - daily seasonality; - Gradient Boosting forecast; - hybrid MLP forecast; - offline Fitted-Q; - random; - oracle (used only as an upper-bound benchmark).

## Key metrics - Regression: MAE, RMSE, R2 for the restriction rate in the next interval. - Classification: ROC-AUC, average precision, precision/recall/F1. - Decision: captured loss, capture vs. total loss, and efficiency vs. top-k oracle.

 ## Usage example

Each row in `results/tables/exemplos_decisao_teste.csv` represents a candidate decision for a specific plant and time step. For operational use, rank the plants for each time step by `score_offline_q`, `score_gb_taxa`, or `score_mlp_hibrido`, and select the top-k based on the monitoring budget.

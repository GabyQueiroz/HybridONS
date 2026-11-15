# HybridONS

Code and results for the study **"Ranking photovoltaic plants by expected constrained-off loss with gradient boosting, learning to rank, and fitted-Q evaluation on public Brazilian operator records"**.

The study uses the public half-hourly records of constrained-off events of photovoltaic (PV) plants published by the Brazilian National System Operator (ONS). It evaluates score functions that order plants by their expected loss in the next half-hour interval, so that an analyst can monitor the *k* plants with the highest expected loss.

## Data

- Source: ONS open data, dataset *Restrição de Operação por Constrained-off de Usinas Fotovoltaicas*, <https://dados.ons.org.br/dataset/restricao_coff_fotovoltaica>.
- Period: 2024-04-01 to 2026-08-25. The data have 2,830,896 half-hourly records from 87 plants or plant groups.
- Files in `data/raw/`: the monthly Parquet files and the ONS data dictionary (JSON and PDF).
- Consolidated table: `data/processed/restricao_coff_fotovoltaica_consolidado.parquet`.

Power values are mean values over the interval (MWmed). Energy is reported in MWh, computed as MWmed × 0.5 h.

## Problem formulation

**Derived loss.**
- The loss is `L = max(G_ref − G, 0)`, where `G_ref` is the reference generation estimated by the operator and `G` is the observed generation.
- The loss splits into two parts, an availability component and a restriction component:
  - availability: `max(G_ref − A, 0)`;
  - restriction: `max(min(G_ref, A) − G, 0)`.
  - Here `A` is the verified availability of the plant.
- Records with and without a restriction code (ENE, CNF, REL) are analysed separately.

**Monitoring task.**
- At the start of each target interval, a policy scores every reporting plant and selects the *k* highest scores, with *k* ∈ {5, 10, 20}.
- The candidate intervals start between 06:00 and 18:30.
- The selection does not change the generation of the plant. Each policy is therefore evaluated by how much of the realized loss the selected plants contain.

**Information protocols.**

| Protocol | Information used | Persistence reference |
|---|---|---|
| A | Records up to interval *t* | `L(t)` |
| B (main) | Records up to *t* − 1, i.e. a one-interval reporting delay | `L(t−1)` |

Lags are computed on a regular 30-min grid per plant, so a gap in the records does not shift them.

## Score functions

- **Heuristics:**
  - random;
  - persistence;
  - recent mean;
  - previous day;
  - plant-hour mean;
  - plant size.
- **Histogram gradient boosting (scikit-learn), four targets:**
  - event (probability of a coded restriction);
  - rate (loss divided by reference generation);
  - loss;
  - discounted return `R(H) = Σ γ^(h−1) L(t+h)`, with H = 6 and γ = 0.92.
- **Fitted-Q:** iterative fitted-Q evaluation with bootstrapped Bellman targets.
- **LightGBM:** loss regression and a LambdaMART ranker that treats each interval as one query.
- **Same target and features, other model families:**
  - XGBoost;
  - CatBoost;
  - an MLP with entity embeddings;
  - a GRU encoder over the last 12 intervals, with multiple seeds.

Because selecting a plant does not change its next state, fitted-Q estimates the expected discounted future loss of each plant. Its ranking therefore converges to the direct regression of the discounted return. The experiments confirm this.

## Evaluation

- Rolling-origin evaluation:
  - expanding training window with quarterly retraining;
  - 6 folds, giving 18 test months from March 2025 to August 2026;
  - the last month before each test period is used for validation;
  - a 6-hour purge separates training from test.
- Original split, used for model families, ablations, horizons and fitted-Q iterations: training until March 2026, validation from April to June 2026, and test in July and August 2026.
- Metrics:
  - captured loss and oracle efficiency (captured loss divided by the loss of the *k* largest realized losses);
  - NDCG@k, precision@k and recall@k;
  - MAE, RMSE, R² and skill relative to persistence.
- Each metric is computed with the loss of the next interval and with the cumulative loss of the next six intervals.
- Uncertainty: day-block bootstrap with 2000 replications and paired differences between policies.

## Main results (rolling origin, Protocol B, k = 10)

| Policy | Oracle efficiency, next interval | Oracle efficiency, next 6 intervals |
|---|---|---|
| Random | 0.253 | 0.264 |
| Plant size | 0.756 | 0.796 |
| Persistence | 0.874 | 0.788 |
| HGB-event | 0.789 | 0.748 |
| HGB-rate | 0.894 | 0.819 |
| HGB-loss | 0.920 | 0.880 |
| LGBM-loss | 0.921 | 0.883 |
| HGB-return (H = 6) | 0.905 | 0.898 |
| Fitted-Q (J = 6) | 0.912 | 0.899 |
| LambdaMART | 0.903 | 0.847 |

**Other findings.**
- HGB-loss exceeded persistence in all 18 test months.
- **Composition of the loss:** records without a restriction code hold 10.2% of the derived loss, and the availability component is 4.6% of it.
- **Information protocol:** with the information of Protocol A, the gain of HGB-loss over persistence falls from 0.046 to 0.015.
- **Model families:** gradient boosting libraries, the MLP and the GRU differ by less than 0.01 in oracle efficiency in the original split.

## Repository structure

```
data/                     raw ONS files and consolidated table
src/ranking/              current experiment pipeline
  common.py               grid, features, targets, protocols, splits, metrics, bootstrap
  models.py               heuristics, gradient boosting targets, fitted-Q, LightGBM, LambdaMART
  run_rolling.py          rolling-origin evaluation (folds 1-6, protocols A and B)
  run_main.py             original split: hyperparameter grid, policies, model families,
                          ablation, horizon and discount grid, fitted-Q iterations, coded losses
  analysis.py             summaries, bootstrap intervals, monthly and subsystem results,
                          onsets, turnover, protocol comparison
  figures.py              vector figures (PDF)
results_ranking/          result tables (CSV); large score files are not versioned
scripts/run_all_experiments.ps1   runs the whole pipeline in sequence
src/hybridons_experiments.py, scripts/run_experiments.py, results/
                          earlier version of the study (kept for comparison)
```

## Reproducing the results

```bash
pip install -r requirements.txt
python -m src.ranking.run_rolling --folds 1,2,3,4,5,6 --protocols B,A
python -m src.ranking.analysis
python -m src.ranking.run_main --stages grid,policies,ablation,horizon,fqi,coded,families
python -m src.ranking.figures results_ranking/figures
```

On Windows, `scripts/run_all_experiments.ps1` runs the same sequence and writes logs to `results_ranking/`. The full run took about four hours on a desktop CPU.

Tested with:

| Package | Version |
|---|---|
| Python | 3.10 |
| scikit-learn | 1.7.2 |
| LightGBM | 4.7.0 |
| XGBoost | 3.2.0 |
| CatBoost | 1.2.10 |
| PyTorch | 2.10 (CPU) |

## Limitations

- The data contain no weather, load, network topology or transmission-limit variables.
- The derived loss depends on the reference generation estimated by the operator.
- The real-time availability of each field is not documented, so Protocol B represents one assumed reporting delay.
- The ranking supports monitoring and does not reduce curtailment by itself.

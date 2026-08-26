# Resultados para o artigo

## Titulo recomendado

**Priorizacao Operacional de Restricoes Constrained-Off em Usinas Fotovoltaicas por Modelo Hibrido de Previsao Temporal e Reinforcement Learning Offline**

## Escopo

Foram usados somente dados abertos da ONS do dataset de restricao de operacao por constrained-off fotovoltaico. O periodo consolidado vai de 2024-04-01 a 2026-08-25, com 2.830.896 registros semi-horarios.

## Principais achados

- 24,47% das linhas possuem restricao codificada.
- A coluna `val_geracaoreferenciafinal` tem cerca de 98,3% de ausencia, entao nao foi usada como alvo principal.
- A perda operacional foi derivada como `max(val_geracaoreferencia - val_geracao, 0)`.
- O maior volume de perda derivada aparece em ENE, seguido por CNF e REL.
- MG, BA, PI, RN e CE concentram a maior parte da perda derivada.

## Modelagem preditiva

Split temporal:

- Treino: ate 2026-03.
- Validacao: 2026-04 a 2026-06.
- Teste: 2026-07 a 2026-08.

Melhor modelo tabular temporal:

- R2 de teste para taxa de restricao no proximo intervalo: 0,7655.
- MAE de teste: 0,0996.
- ROC-AUC de teste para evento de restricao: 0,9820.
- F1 de teste: 0,9441.

MLP neural:

- R2 de teste: 0,6247.
- ROC-AUC de teste: 0,9754.
- F1 de teste: 0,9408.
- A curva de aprendizado apresenta queda consistente de loss e MAE de validacao.

## Politicas de decisao top-10

No periodo de teste, selecionando 10 usinas/conjuntos por instante:

| Politica | Captura vs perda total | Eficiencia vs oraculo |
|---|---:|---:|
| Gradient Boosting taxa | 45,60% | 95,12% |
| Offline fitted-Q | 44,78% | 93,40% |
| MLP hibrido | 44,12% | 92,03% |
| Persistencia | 42,74% | 89,14% |
| Gradient Boosting evento | 39,70% | 82,81% |
| Sazonal diaria | 36,66% | 76,48% |

## Figuras geradas

- `results/figures/curva_aprendizado_mlp.png`
- `results/figures/busca_hiperparametros.png`
- `results/figures/politicas_top10.png`
- `results/figures/robustez_mensal_politicas.png`
- `results/figures/eda_perda_mensal.png`
- `results/figures/eda_perda_estado.png`
- `results/figures/eda_perda_razao.png`
- `results/figures/eda_perfil_intradiario.png`
- `results/figures/eda_top_usinas.png`

## Interpretacao cientifica

O resultado mais forte e que a decisao baseada na taxa prevista pelo Gradient Boosting captura 95,1% do teto superior top-k no teste. O fitted-Q offline tambem e competitivo, com 93,4% do oraculo, mostrando que a formulacao sequencial com retorno futuro descontado e promissora.

Para artigo A1, a contribuicao deve ser descrita como suporte decisorio e priorizacao operacional. Nao se deve afirmar controle fisico direto do constrained-off, porque a base nao contem acoes de controle, topologia da rede, carga, clima, preco ou despacho.

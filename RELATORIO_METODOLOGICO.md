# Analise metodologica - ONS constrained-off fotovoltaico

Fonte usada: dataset publico da ONS "Dados de Restricao de Operacao por Constrained-off de Usinas Fotovoltaicas", baixado via API CKAN em 2026-08-26.

## Dados baixados

- 29 arquivos mensais em Parquet: 2024-04 a 2026-08.
- Dicionario de dados em JSON e PDF.
- Consolidado local: `processed/restricao_coff_fotovoltaica_consolidado.parquet`.
- Total consolidado: 2.830.896 linhas semi-horarias.
- Periodo observado: 2024-04-01 00:00:00 a 2026-08-25 23:30:00.
- Unidades: 87 `id_ons`, 101 nomes de usina/conjunto, 10 estados e 3 subsistemas.

## Variaveis realmente disponiveis

O dataset contem identificacao da usina/conjunto, estado, subsistema, instante semi-horario, geracao, geracao limitada, disponibilidade, geracao de referencia, geracao de referencia final, codigo/origem/descricao da restricao.

Ponto critico: `val_geracaoreferenciafinal` esta ausente em cerca de 98,3% das linhas. Portanto, a variavel operacional mais viavel para modelagem ampla e `val_geracaoreferencia`. A perda derivada mais defensavel, usando apenas este dataset, e:

`perda_ref_mwmed = max(val_geracaoreferencia - val_geracao, 0)`

Essa derivacao deve ser descrita como "perda em relacao a geracao de referencia", nao como medicao fisica causal completa de energia nao gerada.

## Diagnostico inicial

- Linhas com codigo de restricao: 692.579, ou 24,47% do total.
- Perda total derivada pela referencia: 48.961.108 MWmed por intervalo semi-horario.
- Perda em linhas com restricao codificada: 43.964.841 MWmed por intervalo semi-horario.
- Razoes mais relevantes por perda derivada:
  - ENE: 29.576.768
  - CNF: 9.597.031
  - REL: 4.791.035
- Estados com maior perda derivada:
  - MG: 19.835.130
  - BA: 8.433.985
  - PI: 5.782.734
  - RN: 4.397.788
  - CE: 4.160.502

## Baseline preditivo

Foi treinado um baseline temporal com features de calendario, usina, estado, subsistema, valores operacionais atuais e defasagens de 1, 2, 48 e 96 intervalos. O split foi cronologico:

- Treino: 2024-04 a 2026-03, com amostra deterministica de 900.000 linhas.
- Validacao: 2026-04 a 2026-06.
- Teste: 2026-07 a 2026-08.
- Apenas horarios com `val_geracaoreferencia > 0`.

Resultados em teste para prever o proximo intervalo semi-horario:

- Regressao da taxa de restricao:
  - MAE modelo: 0,0986
  - RMSE modelo: 0,1561
  - R2 modelo: 0,7646
  - R2 persistencia: 0,3457
- Classificacao de evento de restricao:
  - Prevalencia no teste: 65,85%
  - ROC-AUC: 0,9827
  - Average precision: 0,9915
  - F1 no limiar 0,5: 0,9413

Conclusao: ha forte previsibilidade temporal para risco e intensidade de constrained-off. Isso sustenta uma abordagem hibrida com modelo preditivo profundo ou ensemble temporal.

## Melhor proposta de artigo

Tema recomendado:

**Modelo hibrido de previsao temporal e reinforcement learning offline para priorizacao operacional de constrained-off fotovoltaico no SIN.**

Pergunta cientifica:

Como usar exclusivamente historicos semi-horarios de restricao fotovoltaica para antecipar eventos de constrained-off e alocar um orcamento limitado de monitoramento/intervencao entre usinas, reduzindo a perda esperada capturada pelo processo decisorio?

Por que esta formulacao e defensavel:

- O dataset nao contem carga, rede eletrica, clima, despacho, preco, ordens do operador nem acoes historicas explicitas. Portanto, ele nao sustenta, sozinho, um RL de controle fisico real do sistema.
- Ele sustenta bem previsao de risco/intensidade e uma decisao sequencial de priorizacao: a cada intervalo, escolher quais usinas/conjuntos devem receber atencao operacional, auditoria, mitigacao ou analise preventiva.
- A recompensa pode ser definida com variaveis observadas: perda futura capturada entre as usinas priorizadas menos custo de priorizacao. Isso evita inventar estados fisicos nao observados.

## Desenho metodologico sugerido

1. Consolidacao e auditoria
   - Usar todos os meses em Parquet.
   - Manter split temporal bloqueado por meses.
   - Tratar `val_geracaoreferenciafinal` como variavel auxiliar, nao alvo principal.

2. Modelo preditivo
   - Alvo 1: probabilidade de restricao no horizonte h = 1, 2, 6, 12 intervalos.
   - Alvo 2: taxa/perda esperada de restricao.
   - Baselines: persistencia, sazonalidade por hora-usina, gradient boosting.
   - Modelos fortes: Temporal Fusion Transformer, TCN, LSTM/GRU com embeddings de usina/estado/subsistema, ou TabTransformer temporal.

3. Ambiente de decisao
   - Estado: historico recente por usina, calendario, subsistema, estado, codigos observados ate o instante atual, previsoes do modelo.
   - Acao: selecionar top-k usinas/conjuntos para priorizacao em cada intervalo.
   - Recompensa observavel: soma da perda futura observada nas usinas selecionadas, normalizada, menos custo por selecao.
   - Restricoes: orcamento k, diversidade por subsistema/estado, penalizacao por excesso de acionamentos.

4. Algoritmos
   - Comecar com contextual bandit e ranking learning-to-rank.
   - Evoluir para offline RL: Conservative Q-Learning, Implicit Q-Learning ou fitted Q evaluation em ambiente de replay historico.
   - Comparar contra heuristicas: top perdas historicas, persistencia, risco previsto isolado, round-robin por estado/subsistema.

5. Avaliacao
   - Captura de perda no top-k.
   - NDCG@k ou recall de eventos severos@k.
   - Reducao de perda nao monitorada em relacao aos baselines.
   - Robustez por estado, subsistema, razao ENE/CNF/REL e periodos fora da amostra.
   - Ablation study: sem lags, sem embeddings de usina, sem previsao DL, sem componente RL.

## O que nao afirmar sem dados externos

- Nao afirmar reducao fisica real de constrained-off no SIN.
- Nao afirmar causalidade entre acao do agente e geracao, porque o dataset nao tem acoes de controle.
- Nao afirmar otimizacao de despacho ou rede eletrica.
- Nao usar clima, irradiancia, carga, preco ou topologia se o requisito for "somente estes dados".

## Caminho para artigo A1

O caminho mais forte e vender a contribuicao como suporte decisorio operacional baseado em dados publicos, com validacao temporal rigorosa e interpretabilidade por tipo de restricao. Para ficar em nivel A1, o artigo deve enfatizar:

- Reprodutibilidade com dados abertos ONS.
- Problema real e brasileiro, ainda pouco explorado para constrained-off fotovoltaico.
- Split temporal honesto e avaliacao por regioes/tipos de restricao.
- Comparacao ampla contra baselines simples e fortes.
- Formulacao cuidadosa de offline RL/ranking sequencial, sem alegar controle fisico nao observavel.

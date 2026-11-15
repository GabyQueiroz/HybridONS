# Runs all experiments in sequence. Logs in results_ranking\log_*.txt
Set-Location (Split-Path $PSScriptRoot -Parent)
python -u -m src.ranking.run_rolling --folds 1,2,3,4,5,6 --protocols B,A *> results_ranking\log_rolling.txt
python -u -m src.ranking.analysis *> results_ranking\log_analysis.txt
python -u -m src.ranking.run_main --stages grid,policies,ablation,horizon,fqi,coded,families *> results_ranking\log_main.txt

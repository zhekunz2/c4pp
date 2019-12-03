#!/usr/bin/env bash
# original + path 5
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep dt_ data_size -saveas plots/wass_avg_static_timeseries_data_ast_5.png &> logs/log_`date +"%Y%m%d%H%M"`
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep g_ -saveas plots/wass_avg_static_timeseries_g_ast_5.png  &> logs/log_`date +"%Y%m%d%H%M"`
# original + full
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast2.csv -l csvs/timeseries_mutants_7_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep dt_ data_size -saveas plots/wass_avg_static_timeseries_data_ast2.png  &> logs/log_`date +"%Y%m%d%H%M"`
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast2.csv -l csvs/timeseries_mutants_7_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep g_ -saveas plots/wass_avg_static_timeseries_g_ast2.png  &> logs/log_`date +"%Y%m%d%H%M"`


./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep dt_ data_size -saveas plots/wass_avg_static_mixture_data_ast_5.png &> logs/log_`date +"%Y%m%d%H%M"`
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep g_ -saveas plots/wass_avg_static_mixture_g_ast_5.png  &> logs/log_`date +"%Y%m%d%H%M"`
# original + full
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast2.csv -l csvs/mixture_mutants_5_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep dt_ data_size -saveas plots/wass_avg_static_mixture_data_ast2.png  &> logs/log_`date +"%Y%m%d%H%M"`
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast2.csv -l csvs/mixture_mutants_5_metrics_wass.csv -a rf -m wass -suf avg  -bw -plt -keep g_ -saveas plots/wass_avg_static_mixture_g_ast2.png  &> logs/log_`date +"%Y%m%d%H%M"`
# convergence rhat_min for prediction on original

./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_min_avg_static_lrm_original.png -keep g_ dt_ var_min var_max data_size -selected generation/lrm_mutants_9_selected.txt -testf csvs/all_original_all_features.csv -testl csvs/lrm_original_metrics.csv &> logs/rhat_min_avg_static_lrm_original.text

./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_min_avg_static_timeseries_original.png -keep g_ dt_ var_min var_max data_size -selected generation/timeseries_mutants_7_selected.txt -testf csvs/all_original_all_features.csv -testl csvs/timeseries_original_metrics.csv &> logs/rhat_min_avg_static_timeseries_original.text

./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_min_avg_static_mixture_original.png -keep g_ dt_ var_min var_max data_size -selected generation/mixture_mutants_5_selected.txt -testf csvs/all_original_all_features.csv -testl csvs/mixture_original_metrics.csv &> logs/rhat_min_avg_static_mixture_original.text

# accuracy wass plots for prediction on original
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m wass -suf avg -bw -plt -saveas plots/wass_avg_static_lrm_original.png -keep g_ dt_ var_min var_max data_size -selected generation/lrm_mutants_9_selected.txt -testf csvs/all_original_all_features.csv -testl csvs/lrm_original_metrics.csv &> logs/wass_avg_static_lrm_original.text

./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a rf -m wass -suf avg -bw -plt -saveas plots/wass_avg_static_timeseries_original.png -keep g_ dt_ var_min var_max data_size -selected generation/timeseries_mutants_7_selected.txt -testf csvs/all_original_all_features.csv -testl csvs/timeseries_original_metrics.csv &> logs/wass_avg_static_timeseries_original.text

./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m wass -suf avg -bw -plt -saveas plots/wass_avg_static_mixture_original.png -keep g_ dt_ var_min var_max data_size -selected generation/mixture_mutants_5_selected.txt -testf csvs/all_original_all_features.csv -testl csvs/mixture_original_metrics.csv &> logs/wass_avg_static_mixture_original.text
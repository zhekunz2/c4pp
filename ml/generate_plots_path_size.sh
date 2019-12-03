#!/usr/bin/env bash
# convergence plot
# ./generate_plots_path_size [batch name]
for size in `echo 2 5 10 20`;
do
./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_$size.csv -l csvs/$1_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_mixture_ast_$size.png -keep g_ dt_ var_min var_max data_size -selected generation/$1_selected.txt &> logs/rhat_avg_static_mixture_ast_$size.txt
./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_$size.csv -l csvs/$1_metrics.csv -a rf -m wass -suf avg -bw -plt -saveas plots/wass_avg_static_mixture_ast_$size.png -keep g_ dt_ var_min var_max data_size -selected generation/$1_selected.txt &> logs/wass_avg_static_mixture_ast_$size.txt
done
#!/usr/bin/env bash
# convergence plot

run(){
prefix=`echo $1 | cut -d"_" -f1`
met_prefix=`echo $1 | cut -d"_" -f1`
echo $1
echo $2

time ./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -bw -plt -saveas plots/${met_prefix}_avg_static_${prefix}.png -keep dt_ var_min var_max data_size -ignore_vi -cv -selected generation/$1_selected.txt &> logs/${met_prefix}_avg_static_${prefix}.txt
# maj
time ./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a maj -m $2 -suf avg -bw -plt -saveas plots/${met_prefix}_avg_static_${prefix}_maj.png -keep dt_ var_min var_max data_size -ignore_vi -selected generation/$1_selected.txt &> logs/${met_prefix}_avg_static_${prefix}_maj.txt

#time ./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -runtime -bw -th 1.1 -saveas plots/rhat_avg_mix_runtime.png -keep dt_ var_min var_max data_size -cv -ignore_vi -selected  generation/mixture_mutants_5_selected.txt &> logs/rhat_avg_mix_runtime.txt

}
export -f run
printf "lrm_mutants_9\ntimeseries_mutants_7\nmixture_mutants_5" | parallel "run {} rhat_min; run {} wass"


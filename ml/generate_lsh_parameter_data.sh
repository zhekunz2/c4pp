#!/usr/bin/env bash

for ((stage=0;stage<101;stage+=5));
do
  for ((bucket=0;bucket<101;bucket+=5));
  do
    i=$stage
    j=$bucket
    if [[ $stage -eq 0 ]]
    then i=1
    fi
    if [[ $bucket -eq 0 ]]
    then j=1
    fi
#    name="../output/mixture_mutants_"$i"_features_ast_"$j"_lsh.csv"
    name="../output/timeseries_mutants_"$i"_features_ast_"$j"_lsh.csv"
#    echo $name
    time python3 train.py -f csvs/lrm_mutants_9_features.csv -fo $name -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -th 1.05 -keep dt_ var_min var_max data_size -ignore_vi -selected generation/lrm_mutants_9_selected.txt
  done
done
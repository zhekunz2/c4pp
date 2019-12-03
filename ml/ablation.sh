#!/usr/bin/env bash
# ./ablation.sh [lrm_mutants_9] [rhat_min|wass]
# convergence plot
prefix=`echo $1 | cut -d"_" -f1`
echo $prefix
# graph
./train.py -f csvs/$1_features.csv -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -bw -plt -saveas plots/$2_avg_static_${prefix}_graph.png -keep g_ -selected generation/$1_selected.txt &> logs/$2_avg_static_${prefix}_graph_abl.txt

# data
./train.py -f csvs/$1_features.csv -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -bw -plt -saveas plots/$2_avg_static_${prefix}_data.png -keep  dt_ var_min var_max data_size -selected generation/$1_selected.txt &> logs/$2_avg_static_${prefix}_data_abl.txt

# ast
./train.py -f  csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -bw -plt -saveas plots/$2_avg_static_${prefix}_ast_only.png -selected generation/$1_selected.txt &> logs/$2_avg_static_${prefix}_ast_only_abl.txt

# data graph
./train.py -f csvs/$1_features.csv  -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -bw -plt -saveas plots/$2_avg_static_${prefix}_data_graph.png -keep g_ dt_ var_min var_max data_size -selected generation/$1_selected.txt &> logs/$2_avg_static_${prefix}_graph_data_graph_abl.txt

# data ast
./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -bw -plt -saveas plots/$2_avg_static_${prefix}_data_ast.png -keep dt_ var_min var_max data_size -selected generation/$1_selected.txt &> logs/$2_avg_static_${prefix}_data_ast_abl.txt

# ast graph
./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a rf -m $2 -suf avg -bw -plt -saveas plots/$2_avg_static_${prefix}_ast_graph.png -keep g_ -selected generation/$1_selected.txt &> logs/$2_avg_static_${prefix}_ast_graph_abl.txt




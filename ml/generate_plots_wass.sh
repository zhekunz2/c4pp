#!/usr/bin/env bash

# accuracy plot
# wass_avg_static_lrm
time ./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m wass -suf avg -plt -bw -saveas plots/wass_avg_static_lrm.png  -keep dt_ var_min var_max data_size  -ignore_vi -cv -selected  generation/lrm_mutants_9_selected.txt  &> logs/wass_avg_static_lrm.txt
# wass_avg_static_lrm_st
#./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m wass -suf avg -plt -st -bw -saveas plots/wass_avg_static_lrm_st.png
# wass_avg_static_lrm_maj
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a maj -m wass -suf avg -plt -bw -saveas plots/wass_avg_static_lrm_maj.png  -keep dt_ var_min var_max data_size -ignore_vi   -selected  generation/lrm_mutants_9_selected.txt  &> logs/wass_avg_static_lrm_maj.txt


# wass_avg_static_ts
./train.py -f csvs/timeseries_mutants_7_features.csv  -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a rf -m wass -suf avg -plt  -bw -saveas plots/wass_avg_static_timeseries.png  -keep dt_ var_min var_max data_size -ignore_vi -cv -selected generation/timeseries_mutants_7_selected.txt  &> logs/wass_avg_static_timeseries.txt
# wass_avg_static_ts_st
#./train.py -f csvs/timeseries_mutants_7_features.csv  -l csvs/timeseries_mutants_7_metrics.csv  -a rf -m wass -suf avg -plt  -bw -st -saveas plots/wass_avg_static_timeseries_st.png -split 0.92
# wass_avg_static_ts_maj
./train.py -f csvs/timeseries_mutants_7_features.csv  -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a maj -m wass -suf avg -plt  -bw -saveas plots/wass_avg_static_timeseries_maj.png  -keep dt_ var_min var_max data_size -ignore_vi  -selected generation/timeseries_mutants_7_selected.txt  &> logs/wass_avg_static_timeseries_maj.txt
#./train.py -f csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a rf -m wass -suf avg -plt  -bw -saveas plots/wass_avg_static_timeseries_path2.png

# wass_avg_static_mix
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m wass -suf avg -plt -bw -saveas plots/wass_avg_static_mix.png  -keep dt_ var_min var_max data_size  -ignore_vi -cv -selected  generation/mixture_mutants_5_selected.txt &> logs/wass_avg_static_mix.txt
# wass_avg_static_mix_st
#./train.py -f csvs/mixture_mutants_5_features.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m wass -suf avg -plt -bw -st -saveas plots/wass_avg_static_mix_st.png -split 0.92
# wass_avg_static_mix_maj
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a maj -m wass -suf avg -plt -bw -saveas plots/wass_avg_static_mix_maj.png  -keep dt_ var_min var_max data_size -ignore_vi  -selected  generation/mixture_mutants_5_selected.txt &> logs/wass_avg_static_mix_maj.txt

# runtime plots

# accuracy

# wass_avg_static_lrm_runtime
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m wass -th 0.2 -bw -suf avg -runtime -saveas plots/wass_avg_lrm_runtime.png  -keep dt_ var_min var_max data_size -ignore_vi -cv -selected  generation/lrm_mutants_9_selected.txt &> logs/wass_avg_lrm_runtime.txt

# wass_avg_static_ts_runtime
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv  -l csvs/timeseries_mutants_7_metrics.csv -a rf -m wass -suf avg -bw -runtime -th 0.2 -saveas plots/wass_avg_timeseries_runtime.png  -keep dt_ var_min var_max data_size -ignore_vi -cv -selected generation/timeseries_mutants_7_selected.txt &> logs/wass_avg_timeseries_runtime.txt

# wass_avg_static_mix_runtime
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m wass -suf avg -runtime -bw -th 0.2 -saveas plots/wass_avg_mix_runtime.png  -keep dt_ var_min var_max data_size -ignore_vi -cv -selected  generation/mixture_mutants_5_selected.txt &> logs/wass_avg_mix_runtime.txt


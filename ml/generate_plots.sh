#!/usr/bin/env bash
# convergence plot

# rhat_avg_static_lrm

time python3 train.py -f csvs/lrm_mutants_9_features.csv -fo ../output/mixture_mutants_50_features_ast_100_lsh.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -th 1.05 -keep dt_ var_min var_max data_size -ignore_vi -selected generation/lrm_mutants_9_selected.txt
time python3 train.py -f csvs/lrm_mutants_9_features.csv -fo ../mixture_mutants_5_features_ast_5_raw.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -th 1.05 -keep dt_ var_min var_max data_size -ignore_vi -selected generation/lrm_mutants_9_selected.txt

# rhat_avg_static_lrm_st
#./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -st -split 0.92  -saveas plots/rhat_avg_static_lrm_st.png
# rhat_avg_static_lrm_maj
time ./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a maj -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_lrm_maj.png -keep dt_ var_min var_max data_size -ignore_vi -selected generation/lrm_mutants_9_selected.txt &> logs/rhat_avg_static_lrm_maj.txt
#
#./train.py -f csvs/lrm_mutants_9_features_g.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_lrm_path.png

#./train.py -f csvs/lrm_mutants_9_features_g2.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_lrm_path2.png

# rhat_avg_static_timeseries
time ./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a rf -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries.png -keep dt_ var_min var_max data_size -cv -ignore_vi -selected generation/timeseries_mutants_7_selected.txt &> logs/rhat_avg_static_timeseries.txt
# rhat_avg_static_timeseries_st
#./train.py -f csvs/timeseries_mutants_7_features.csv  -l csvs/timeseries_mutants_7_metrics.csv  -a rf -m rhat_min -suf avg -plt -st -split 0.92 -bw -saveas plots/rhat_avg_static_timeseries_st.png
# rhat_avg_static_timeseries_maj
time ./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a maj -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries_maj.png -keep dt_ var_min var_max data_size  -ignore_vi -selected generation/timeseries_mutants_7_selected.txt &> logs/rhat_avg_static_timeseries_maj.txt
# rhat_avg_static_timeseries_path
#./train.py -f csvs/timeseries_mutants_7_features_g.csv  -l csvs/timeseries_mutants_7_metrics.csv  -a rf -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries_path.png
# rhat_avg_static_timeseries_path2
#./train.py -f csvs/timeseries_mutants_7_features_g2.csv  -l csvs/timeseries_mutants_7_metrics.csv  -a rf -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries_path2.png

# rhat_avg_static_mix
time ./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix.png -keep dt_ var_min var_max data_size -cv -ignore_vi -selected generation/mixture_mutants_5_selected.txt  &> logs/rhat_avg_static_mix.txt
# rhat_avg_static_mix_st
#./train.py -f csvs/mixture_mutants_5_features.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -st -split 0.92 -saveas plots/rhat_avg_static_mix_st.png
# rhat_avg_static_mix_maj
time ./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a maj -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix_maj.png -keep dt_ var_min var_max data_size -ignore_vi -selected generation/mixture_mutants_5_selected.txt &> logs/rhat_avg_static_mix_maj.txt
# rhat_avg_static_mix_path
#./train.py -f csvs/mixture_mutants_5_features_g.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix_path.png
# rhat_avg_static_mix_path2
#./train.py -f csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix_path2.png

# runtime plots

# convergence

# rhat_avg_static_lrm_runtime
time ./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -th 1.1 -bw -suf avg -runtime -saveas plots/rhat_avg_lrm_runtime.png -keep dt_ var_min var_max data_size -cv -ignore_vi -selected  generation/lrm_mutants_9_selected.txt &> logs/rhat_avg_lrm_runtime.txt

# rhat_avg_static_timeseries_runtime
time ./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv  -l csvs/timeseries_mutants_7_metrics.csv -a rf -m rhat_min -suf avg -bw -runtime -th 1.1 -saveas plots/rhat_avg_ts_runtime.png -cv -keep dt_ var_min var_max data_size -ignore_vi -selected generation/timeseries_mutants_7_selected.txt  &> logs/rhat_avg_ts_runtime.txt

# rhat_avg_static_mix_runtime
time ./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -runtime -bw -th 1.1 -saveas plots/rhat_avg_mix_runtime.png -keep dt_ var_min var_max data_size -cv -ignore_vi -selected  generation/mixture_mutants_5_selected.txt &> logs/rhat_avg_mix_runtime.txt
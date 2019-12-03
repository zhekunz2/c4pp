#!/usr/bin/env bash
# convergence plot
# ./generate_table_iters.sh [lrm_mutants_9] [lrm]
# rhat_avg_static_lrm
for iter in `seq 100 100 900`; do
./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics_${iter}.csv -a rf -m rhat_min -suf avg -bw -keep g_ dt_ var_min var_max data_size -selected generation/$1_selected.txt -th 1.1 &> logs/rhat_avg_static_$2_${iter}.txt
./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics_${iter}.csv -a rf -m wass -suf avg -bw -keep g_ dt_ var_min var_max data_size -selected generation/$1_selected.txt -th 0.2 &> logs/wass_avg_static_$2_${iter}.txt
done
exit 1
# rhat_avg_static_lrm_st
#./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -st -split 0.92  -saveas plots/rhat_avg_static_lrm_st.png
# rhat_avg_static_lrm_maj
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a maj -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_lrm_maj.png -keep g_ dt_ var_min var_max data_size -selected generation/lrm_mutants_9_selected.txt &> logs/rhat_avg_static_lrm_maj.txt
#
#./train.py -f csvs/lrm_mutants_9_features_g.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_lrm_path.png

#./train.py -f csvs/lrm_mutants_9_features_g2.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_lrm_path2.png

# rhat_avg_static_timeseries
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a rf -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries.png -keep g_ dt_ var_min var_max data_size -selected generation/timeseries_mutants_7_selected.txt &> logs/rhat_avg_static_timeseries.txt
# rhat_avg_static_timeseries_st
#./train.py -f csvs/timeseries_mutants_7_features.csv  -l csvs/timeseries_mutants_7_metrics.csv  -a rf -m rhat_min -suf avg -plt -st -split 0.92 -bw -saveas plots/rhat_avg_static_timeseries_st.png
# rhat_avg_static_timeseries_maj
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l csvs/timeseries_mutants_7_metrics.csv -a maj -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries_maj.png -keep g_ dt_ var_min var_max data_size -selected generation/timeseries_mutants_7_selected.txt &> logs/rhat_avg_static_timeseries_maj.txt
# rhat_avg_static_timeseries_path
#./train.py -f csvs/timeseries_mutants_7_features_g.csv  -l csvs/timeseries_mutants_7_metrics.csv  -a rf -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries_path.png
# rhat_avg_static_timeseries_path2
#./train.py -f csvs/timeseries_mutants_7_features_g2.csv  -l csvs/timeseries_mutants_7_metrics.csv  -a rf -m rhat_min -suf avg -plt  -bw -saveas plots/rhat_avg_static_timeseries_path2.png

# rhat_avg_static_mix
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix.png -keep g_ dt_ var_min var_max data_size -selected generation/mixture_mutants_5_selected.txt  &> logs/rhat_avg_static_mix.txt
# rhat_avg_static_mix_st
#./train.py -f csvs/mixture_mutants_5_features.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -st -split 0.92 -saveas plots/rhat_avg_static_mix_st.png
# rhat_avg_static_mix_maj
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a maj -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix_maj.png -keep g_ dt_ var_min var_max data_size -selected generation/mixture_mutants_5_selected.txt &> logs/rhat_avg_static_mix_maj.txt
# rhat_avg_static_mix_path
#./train.py -f csvs/mixture_mutants_5_features_g.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix_path.png
# rhat_avg_static_mix_path2
#./train.py -f csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -saveas plots/rhat_avg_static_mix_path2.png

# runtime plots

# convergence

# rhat_avg_static_lrm_runtime
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l csvs/lrm_mutants_9_metrics.csv -a rf -m rhat_min -th 1.1 -bw -suf avg -runtime -saveas plots/rhat_avg_lrm_runtime.png -keep g_ dt_ var_min var_max data_size -selected  generation/lrm_mutants_9_selected.txt &> logs/rhat_avg_lrm_runtime.txt

# rhat_avg_static_timeseries_runtime
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv  -l csvs/timeseries_mutants_7_metrics.csv -a rf -m rhat_min -suf avg -bw -runtime -th 1.1 -saveas plots/rhat_avg_ts_runtime.png -keep g_ dt_ var_min var_max data_size -selected generation/timeseries_mutants_7_selected.txt  &> logs/rhat_avg_ts_runtime.txt

# rhat_avg_static_mix_runtime
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -runtime -bw -th 1.1 -saveas plots/rhat_avg_mix_runtime.png -keep g_ dt_ var_min var_max data_size -selected  generation/mixture_mutants_5_selected.txt &> logs/rhat_avg_mix_runtime.txt
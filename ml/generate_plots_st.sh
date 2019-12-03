#!/usr/bin/env bash
# convergence plot
# ./generate_plots_st.sh [categories] [prefix] lrm_mutants_9?
plot(){
prefix=`echo $1 | cut -d"_" -f1`
donest=`cat done_st | grep -w "$2" | wc -l`

if [ ${donest} -gt 0 ];then
echo "$2"
return
fi
echo $2
time ./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a rf -m rhat_min -suf avg -bw -plt -saveas plots/rhat_avg_static_${prefix}_st_$2.png -keep dt_ var_min var_max data_size -selected generation/$1_selected.txt -st -tname $2 -cv -ignore_vi &> logs/rhat_avg_static_${prefix}_$2_cv.txt
time ./train.py -f csvs/$1_features.csv -fo csvs/$1_features_ast_5.csv -l csvs/$1_metrics.csv -a rf -m wass -suf avg -bw -plt -saveas plots/wass_avg_static_${prefix}_st_$2.png -keep dt_ var_min var_max data_size -selected generation/$1_selected.txt -st -tname $2  -cv -ignore_vi &> logs/wass_avg_static_${prefix}_$2_cv.txt
echo $prefix
}
export -f plot
cat $1 | grep  -v "\[\|\]\|\{\|\}" | grep -o "[a-zA-Z0-9_.-]\+" |  parallel -j 4 plot $2 {}

#printf "lightspeed\nwells\ndogs_log" | parallel -j 1 plot "lrm_mutants_9" {}
#printf "schools\nradon" | parallel -j 1 plot "lrm_mutants_9" {}
#printf "arma11\ngp-fit\nnile" | parallel -j 1 plot "timeseries_mutants_7" {}
#printf "hmm-sufficient\nma2" | parallel -j 1 plot "timeseries_mutants_7" {}
#printf "gauss_mix\nM0\nNmix0" | parallel -j 1 plot "mixture_mutants_5" {}
#printf "lda\nSurvey" | parallel -j 1 plot "mixture_mutants_5" {}
exit 1







# rhat_avg_static_mix_st
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -st -tname Nmix0 -saveas plots/rhat_avg_static_mix_st_Nmix0.png
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -st -tname M0 -saveas plots/rhat_avg_static_mix_st_M0.png
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics.csv -a rf -m rhat_min -suf avg -plt -bw -st -tname gauss_mix -saveas plots/rhat_avg_static_mix_st_gauss_mix.png


# accuracy


# wass_avg_static_lrm_st
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_g2.csv -l csvs/lrm_mutants_9_metrics_wass.csv -a rf -m wass -suf avg -bw -plt -st -saveas plots/wass_avg_static_lrm_st_lightspeed.png -tname lightspeed
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_g2.csv -l csvs/lrm_mutants_9_metrics_wass.csv -a rf -m wass -suf avg -bw -plt -st -saveas plots/wass_avg_static_lrm_st_wells.png -tname wells
./train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_g2.csv -l csvs/lrm_mutants_9_metrics_wass.csv -a rf -m wass -suf avg -bw -plt -st -saveas plots/wass_avg_static_lrm_st_dogs_log.png -tname dogs_log


# wass_avg_static_timeseries_st
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_g2.csv -l csvs/timeseries_mutants_7_metrics_wass.csv  -a rf -m wass -suf avg -plt -st -bw -tname gp-fit -saveas plots/wass_avg_static_timeseries_st_gp_fit.png
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_g2.csv -l csvs/timeseries_mutants_7_metrics_wass.csv  -a rf -m wass -suf avg -plt -st -bw -tname arma11 -saveas plots/wass_avg_static_timeseries_st_arma11.png
./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_g2.csv -l csvs/timeseries_mutants_7_metrics_wass.csv  -a rf -m wass -suf avg -plt -st -bw -tname nile -saveas plots/wass_avg_static_timeseries_st_nile.png


# wass_avg_static_mix_st
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics_wass.csv -a rf -m wass -suf avg -plt -bw -st -tname Nmix0 -saveas plots/wass_avg_static_mix_st_Nmix0.png
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics_wass.csv -a rf -m wass -suf avg -plt -bw -st -tname M0 -saveas plots/wass_avg_static_mix_st_M0.png
./train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_g2.csv -l csvs/mixture_mutants_5_metrics_wass.csv -a rf -m wass -suf avg -plt -bw -st -tname gauss_mix -saveas plots/wass_avg_static_mix_st_gauss_mix.png


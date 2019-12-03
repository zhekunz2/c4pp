#!/usr/bin/env bash
python combine.py results/w-cv/ex/rhat_avg_static_lrm.png.txt results/wocv/rhat_avg_static_lrm_maj.png.txt c2v/new/results_lrm_mutants_9_rhat.txt rhat_min  plots/rhat_avg_static_lrm.png

python combine.py results/w-cv/ex/rhat_avg_static_timeseries.png.txt results/wocv/rhat_avg_static_timeseries_maj.png.txt c2v/new/results_timeseries_mutants_7_rhat.txt rhat_min plots/rhat_avg_static_timeseries.png
python combine.py results/w-cv/ex/rhat_avg_static_mix.png.txt results/wocv/rhat_avg_static_mix_maj.png.txt c2v/new/results_mixture_mutants_5_rhat.txt rhat_min plots/rhat_avg_static_mix.png
python combine.py results/w-cv/ex/wass_avg_static_mix.png.txt results/wocv/wass_avg_static_mix_maj.png.txt c2v/new/results_mixture_mutants_5_wass.txt wass plots/wass_avg_static_mix.png
python combine.py results/w-cv/ex/wass_avg_static_timeseries.png.txt results/wocv/wass_avg_static_timeseries_maj.png.txt c2v/new/results_timeseries_mutants_7_wass.txt wass plots/wass_avg_static_timeseries.png
python combine.py results/w-cv/ex/wass_avg_static_lrm.png.txt results/wocv/wass_avg_static_lrm_maj.png.txt c2v/new/results_lrm_mutants_9_wass.txt wass  plots/wass_avg_static_lrm.png

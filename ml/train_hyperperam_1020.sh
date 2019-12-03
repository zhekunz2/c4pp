#!/usr/bin/env bash
#for f in `ls csvs/progs20190622-171958263444_metrics_out_iter_*`;
#for f in `ls csvs/progs20190621-205419688528_metrics_out_iter_*`;
(for f in `ls csvs_iter/timeseries_mutants_7_metrics_out_iter_*00`;
do
  iter=`echo $f | cut -d"_" -f8`
  printf "$iter,"
  #python3.6 train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size | grep ">>>"
  #out=`python3.6 train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l $f  -a rf -m rhat_min -suf avg -bw -th 1.1 -keep g_ dt_ var_min var_max data_size`
  out=`./train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size`
  # echo "$out" | grep "common" -A2 -i >../metrics/res/timeseries_mutants_7_iter.progs2
  echo "$out" | grep ">>>>" | cut -d" " -f2
  printf "$iter,"
  echo "$out" | grep "<<<<" | cut -d" " -f2
done) > ../metrics/res/timeseries_mutants_7_iter_wass.res

# (for f in `ls csvs_iter/lrm_mutants_9_metrics_out_iter_*00`;
# do
#   iter=`echo $f | cut -d"_" -f8`
#   printf "$iter,"
#   #python3.6 train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size | grep ">>>"
#   out=`python3.6 train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size`
#   #out=`python3.6 train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size`
#   echo "$out" | grep ">>>>" | cut -d" " -f2
#   printf "$iter,"
#   echo "$out" | grep "<<<<" | cut -d" " -f2
# done) > ../metrics/res/lrm_mutants_9_iter_wass.res

# (for f in `ls csvs_iter/mixture_mutants_5_metrics_out_iter_*00`;
# do
#   iter=`echo $f | cut -d"_" -f8`
#   printf "$iter,"
#   #python3.6 train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size | grep ">>>"
#   out=`python3.6 train.py -f csvs/mixture_mutants_5_features.csv -fo csvs/mixture_mutants_5_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size`
#   #out=`python3.6 train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size`
#   echo "$out" | grep ">>>>" | cut -d" " -f2
#   printf "$iter,"
#   echo "$out" | grep "<<<<" | cut -d" " -f2
# done) > ../metrics/res/mixture_mutants_5_iter_wass.res

# (for f in `ls csvs/progs20190621-205419688528_metrics_out_iter_*`;
# do
#   iter=`echo $f | cut -d"_" -f5`
#   printf "$iter "
#   #python3.6 train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size | grep ">>>"
#   out=`python3.6 train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size`
#   echo "$out" | grep ">>>>" | cut -d" " -f2
#   printf "$iter "
#   echo "$out" | grep "<<<<" | cut -d" " -f2
# done) > progs20190621-205419688528.res
# 
# (for f in `ls csvs/progs20190621-205423791989_metrics_out_iter_*`;
# do
#   iter=`echo $f | cut -d"_" -f5`
#   printf "$iter "
#   #python3.6 train.py -f csvs/lrm_mutants_9_features.csv -fo csvs/lrm_mutants_9_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size | grep ">>>"
#   out=`python3.6 train.py -f csvs/timeseries_mutants_7_features.csv -fo csvs/timeseries_mutants_7_features_ast_5.csv -l $f  -a rf -m wass -suf avg -bw -th 0.2 -keep g_ dt_ var_min var_max data_size`
#   echo "$out" | grep ">>>>" | cut -d" " -f2
#   printf "$iter "
#   echo "$out" | grep "<<<<" | cut -d" " -f2
# done) > progs20190621-205423791989.res

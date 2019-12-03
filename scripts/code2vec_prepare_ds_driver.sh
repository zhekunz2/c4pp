#!/usr/bin/env bash

#./code2vec_prepare_ds.sh ../programs/timeseries_mutants_7/ ../ml/generation/timeseries_mutants_7_final.txt 1
#./code2vec_prepare_ds.sh ../programs/timeseries_mutants_7/ ../ml/generation/timeseries_mutants_7_final.txt 2
#./code2vec_prepare_ds.sh ../programs/timeseries_mutants_7/ ../ml/generation/timeseries_mutants_7_final.txt 3

./code2vec_prepare_ds.sh ../programs/lrm_mutants_9/ ../ml/generation/lrm_mutants_9_final.txt 1
./code2vec_prepare_ds.sh ../programs/lrm_mutants_9/ ../ml/generation/lrm_mutants_9_final.txt 2
./code2vec_prepare_ds.sh ../programs/lrm_mutants_9/ ../ml/generation/lrm_mutants_9_final.txt 3

./code2vec_prepare_ds.sh ../programs/mixture_mutants_5/ ../ml/generation/mixture_mutants_5_final.txt 1
./code2vec_prepare_ds.sh ../programs/mixture_mutants_5/ ../ml/generation/mixture_mutants_5_final.txt 2
./code2vec_prepare_ds.sh ../programs/mixture_mutants_5/ ../ml/generation/mixture_mutants_5_final.txt 3

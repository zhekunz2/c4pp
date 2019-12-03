#! /usr/bin/env bash
# ./pre_run_metrics.sh stan_install_dir curr_model_name iter

metrics_file_path=/srv/local/scratch/lrm_mutant/metrics_0301.py
#stan_install_dir=${1%/*}
stan_install_dir=${1}
curr_model_name=${2%/*}
curr_model_name=${2#*/}
#iter=$3
if [ -z "$1" ] | [ -z "$2" ] ; then
   echo "usage: ./pre_run_metrics.sh stan_install_dir curr_model_name"
   exit 0
fi
ref=100000
min=1000
if [ ! -f  $stan_install_dir/$curr_model_name/output_$ref.csv ]; then
    echo "file doesn't exist: $stan_install_dir/$curr_model_name/output_$ref.csv"
    exit 0
fi
if [ ! -f  $stan_install_dir/$curr_model_name/output_$min.csv ]; then
    sed '/^#/ d' $stan_install_dir/$curr_model_name/output_$ref.csv > $stan_install_dir/$curr_model_name/output_${ref}_temp.csv
    rm_iter=$((100000-min))
    head -n -$rm_iter $stan_install_dir/$curr_model_name/output_${ref}_temp.csv > $stan_install_dir/$curr_model_name/output_$min.csv
fi

#if [ ! -f  $stan_install_dir/$curr_model_name/output_${ref}_cut.csv ]; then
#    tail -2000 $stan_install_dir/$curr_model_name/output_${ref}_temp.csv > $stan_install_dir/$curr_model_name/output_${ref}_cut.csv
#fi
#$stan_install_dir/bin/stansummary $stan_install_dir/$curr_model_name/output_$ref.csv | awk '/StdDev/{f=1}/Samples/{f=0}f' | head -n -1 > summary_$ref
#$stan_install_dir/bin/stansummary $stan_install_dir/$curr_model_name/output_$min.csv | awk '/StdDev/{f=1}/Samples/{f=0}f' | head -n -1 > summary_$min
#conv_ref=$($metrics_file_path -c -fs summary_$ref -m rhat)
#conv_min=$($metrics_file_path -c -fs summary_$min -m rhat)
cd $stan_install_dir/$curr_model_name/
if [ -f rw_summary_100000 ]; then rm rw_summary_100000; fi
if [ -f rw_summary_1000 ]; then rm rw_summary_1000; fi
$stan_install_dir/bin/stansummary output_100000.csv --csv_file=rw_summary_100000 > /dev/null
if [ ! -f rw_summary_100000 ]; then echo "$curr_model_name no samples"; exit 0; fi
conv_ref=$($metrics_file_path -c -fs rw_summary_100000 -m rhat)
$stan_install_dir/bin/stansummary output_1000.csv --csv_file=rw_summary_1000 > /dev/null
if [ ! -f rw_summary_1000 ]; then echo "$curr_model_name no samples"; exit 0; fi
conv_min=$($metrics_file_path -c -fs rw_summary_1000 -m rhat)
accu=$($metrics_file_path -fc $stan_install_dir/$curr_model_name/output_${ref}.csv -fc $stan_install_dir/$curr_model_name/output_$min.csv -m t -m ks -m kl -m smkl -m hell)
printf "%s,%s,%s,%s\n" $curr_model_name "$conv_ref" "$conv_min" "$accu" | sed 's/ //g' | sed 's/\///g'

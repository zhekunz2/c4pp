#!/usr/bin/env bash
# ./run_metrics.sh metrics/metrics.py programs/vols/ 5100

metric_file=$1
source_dir=$2
algo="vb"
algo_ref="nuts"
iter=$3
metric=""
csv_file_name=${2//\//_}
[ "${csv_file_name: -1}" == "_" ] && csv_file_name=${csv_file_name::-1}
csv_file_name=${csv_file_name}_${algo}_${iter}_$metric
echo "writing to $csv_file_name.csv"
if [ -e $csv_file_name.csv ]; then
    rm $csv_file_name.csv
fi

for chap in `find $2 -type d`; do
    #if [ ! -e "$chap/model.stan" ]; then echo "skip $chap"; continue; fi
    if [ ! -e "$chap/model.stan" ]; then continue; fi
    chap_name=${chap##*/}
    param_file="param_${algo}_${iter}n"
    param_file_ref="param_${algo_ref}_${iter}n"
    #if [ ! -e "$chap/$param_file" ]; then echo "skip $chap/$param_file"; continue; fi #| tee $csv_file_name.log
    if [ ! -e "$chap/param_${algo}_${iter}" ]; then  continue; fi #| tee $csv_file_name.log
    if [ ! -e "$chap/param_${algo_ref}_${iter}" ]; then  continue; fi #| tee $csv_file_name.log
    echo $chap
    sed -E 's/([[:digit:]]),([[:digit:]])/\1_\2/' $chap/param_${algo}_${iter} > $chap/param_${algo}_${iter}n
    sed -E 's/([[:digit:]]),([[:digit:]])/\1_\2/' $chap/param_${algo_ref}_${iter} > $chap/param_${algo_ref}_${iter}n
    ./$metric_file $chap/$param_file $chap/$param_file_ref $metric | sed -e "s/^/$chap_name,/" >> $csv_file_name.csv
done

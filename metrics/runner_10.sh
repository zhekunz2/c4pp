#!/usr/bin/env bash

# ./runner_10.sh programs/vols/ programs_vols_vb_5100_.csv
# find azure_batch/csvs/csv1115_2/download/export/programs/ -maxdepth 1 -mindepth 1 -type d | xargs -n 1 -P 12 -I {} sh -c "./runner_10.sh {} azure_batch/csvs/csv1115_2/reduce_metric.csv"

counter=0
feature_file=feature.py
source_dir=$1
result_file=$2
#file_name=$(basename "${result_file%.*}")
file_name=${1//\//_}
output_file="feature_${file_name}.csv"
echo $output_file

echo "program" >  $output_file
for chap in `find $source_dir -type d`; do
    #echo $chap
    if [ ! -e "$chap/model.stan" ]; then  continue; fi #echo "skip $chap";
    stan_file=$chap/model.stan
    data_file=$chap/data.json
    ./$feature_file $stan_file $result_file $data_file $output_file
    counter=$((counter+1))
    if [ $(( $counter % 500 )) == 1 ]; then
        echo $counter
    fi
    #if [[ $counter -gt 10 ]]; then break; fi
done

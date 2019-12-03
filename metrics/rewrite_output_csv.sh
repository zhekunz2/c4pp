#!/usr/bin/env bash

for i in 1 2 3 4; do
    sed '/^#/ d' output_1000_$i.csv > output_1000_temp_$i.csv
done
for iter in `seq 100 100 1000`; do
    rm_iter=$((1000-iter))

    for i in 1 2 3 4; do
        head -n -$rm_iter output_1000_temp_$i.csv > output_temp_$i.csv
        sed -i '1,1000d' output_temp_$i.csv
    done

    rm rw_summary_temp
    /scratch/lrm_mutant/cmdstan/stansummary output_temp_*.csv --csv_file=rw_summary_temp > /dev/null
    ret=$(./metrics_0301.py -c -fs rw_summary_temp -m rhat)
    echo $iter,$ret
done

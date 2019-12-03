#!/usr/bin/env bash

for i in 1 2 3 4; do
    zcat output_1000_$i.gz | sed '/^#/ d' | sed '2,1001d' > output_1000_temp_$i.csv
done
for iter in `seq 100 100 1000`; do
    rm_iter=$((iter+2))

    for i in 1 2 3 4; do
        sed ''"$rm_iter"',1001d' output_1000_temp_$i.csv > output_temp_$i.csv
    done

    rm rw_summary_temp
    /home/zixin/Documents/are/PPVM/trans_runner/cmdstan-2.16.0/bin/stansummary output_temp_*.csv --csv_file=rw_summary_temp > /dev/null
    ret=$(~/Documents/are/c4pp/c4pp/metrics/metrics_0301.py -c -fs rw_summary_temp -m rhat)
    echo $iter,$ret
done

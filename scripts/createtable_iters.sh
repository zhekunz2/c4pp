#!/usr/bin/env bash
printf "TimeSeries&"
(printf "%0.02f&" `grep "F1" /home/saikat/projects/c4pp/ml/logs/wass_avg_static_timeseries_*00.txt | cut -d"=" -f2`)
echo "\\\\"
printf "Regression&"
(printf "%0.02f&" `grep "F1" /home/saikat/projects/c4pp/ml/logs/wass_avg_static_lrm_*00.txt | cut -d"=" -f2 `)
echo "\\\\"
printf "Mixture Models&"
(printf "%0.02f&" `grep "F1" /home/saikat/projects/c4pp/ml/logs/wass_avg_static_mixture_*00.txt | cut -d"=" -f2`)
echo "\\\\"
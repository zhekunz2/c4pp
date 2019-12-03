#!/usr/bin/env bash

for pp in `ls progs*/ -d `; do 
    cd $pp; 
    num=$(~/c4pp/metrics/rm_unuse.sh param_use rw_summary_1000 | wc -l)
    rhat=$(~/c4pp/metrics/rm_unuse.sh param_use rw_summary_1000 | cut -d\" -f3 | cut -d, -f10 | sed '/nan/d' | awk '{ total += $1; count++ } END { print total/count }') 
    rhat_ref=$(~/c4pp/metrics/rm_unuse.sh param_use rw_summary_100000 | cut -d\" -f3 | cut -d, -f10 | sed '/nan/d' | awk '{ total += $1; count++ } END { print total/count }') 
    tosize=$(grep -oh "iter=([[:digit:]]*" rw_summary_100000 | sed 's/iter=(//')
    echo $rhat,$rhat_ref,$tosize > rhat
    rhatnew=$(awk -F, '{ if (($3 == 100000) || ($2 <= 1.1)) {print $1}}' rhat)
    rhatnewlabel=$(awk -v var="$rhatnew" 'BEGIN { if (var < 1.01) {print "True"} else {print "False"}}')
    if [ ! -z "$rhatnew" ]; then
        wass=$(cat metrics_out_unuse_0625)
        echo $rhatnewlabel,$rhatnew,$wass,$num > metrics_join_0707
    fi
    cd ..; done
grep "," */metrics_join_0707 | sed 's/\/metrics_join_0707:/,/' > metrics_join_0707_galeb
sed -i '1 i\program,rhat_min_result_avg,rhat_min_value_avg,wass_result_avg,wass_value_avg,wass_result_ext,wass_value_ext,param' metrics_join_0707_galeb
sed -i "/,,/d" metrics_join_0707_galeb


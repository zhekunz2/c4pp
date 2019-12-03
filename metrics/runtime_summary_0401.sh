#!/usr/bin/env bash

total=$(cat progs_curr_dir | wc -l)
cat progs_curr_dir | nl  | xargs -P8 -n1 -I{} sh -c "\
    nn=\$(echo {} | sed 's/ .*$//')
    ss=\$(echo {} | sed 's/^.* //')
    if [ ! -d \$ss ]; then exit 0; fi
    cd \$ss
    echo \$ss \$nn/$total
    timeout 5m time ~/c4pp/metrics/runtime_summary.py > rt_0401 2> rt_time
    "
    # > rerun_metrics_out_0401;\

        #cat */rerun_metrics_out_0401

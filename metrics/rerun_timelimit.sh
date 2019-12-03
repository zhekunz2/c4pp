#!/usr/bin/env bash
# ./rerun_timelimit.sh programs/vols/

counter=0
source_dir=$1
algo="vb"
iter="2100"

for chap in `find $1 -type d`; do
    if [ ! -e "$chap/model.stan" ]; then echo "skip $chap"; continue; fi
    chap_name=${chap##*/}
    cd $chap
    echo $chap
    param_file="param_${algo}_${iter}n"
    if [ -e "$param_file" ]; then echo "skip $chap/$param_file"; cd -;  continue;
    else
        timeout 5m python driver.py sampling 2100 &> output_${algo}_${iter}
        if [ -e "$param_file" ]; then
            sed -E 's/([[:digit:]]),([[:digit:]])/\1_\2/' param_${algo}_${iter} > param_${algo}_${iter}n
        fi
        counter=$((counter+1))
    fi
    cd -
done

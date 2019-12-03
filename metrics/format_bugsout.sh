#!/usr/bin/env bash

# run `~/OpenBUGS-3.2.3/bin/bin/OpenBUGS < script.txt | ./format_bugsout.sh`
grep -A 100 'Rhat' $1 | sed -e '/^$/,$d' | awk -F" " '{print  $1 ", normal," $2,"," $3}' | tail -n +2 > bugs_out
# awk -F" " '{print  $1 ", normal," $2, "," $3}' $1 | tail -n +2 > bugs_out

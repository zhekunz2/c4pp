#!/usr/bin/env bash
grep "(Total)" pro*/stanout_1000_1 | sed 's/\/stanout.*]\s*/,/' | cut -d" " -f1 > mm5_time_1000
sed -i '1 i\program,time' mm5_time_1000
./sum_time.py mm5_tfpn.csv mm5_time_1000

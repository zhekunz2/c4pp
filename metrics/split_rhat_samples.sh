#!/usr/bin/env bash
cat prog*/rhat_unuse_sample > rhat_unuse_sample_all
for ii in `seq 100 100 1000`; do grep "^[^,]*,${ii}," rhat_unuse_sample_all > rhat_unuse_sample_${ii}; done
for ii in `seq 100 100 1000`; do sed -i '1 i\program,iter,rhat_result_avg,rhat_value_avg,rhat_result_ext,rhat_value_ext' rhat_unuse_sample_${ii}; done

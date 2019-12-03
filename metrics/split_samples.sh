#!/usr/bin/env bash
cat prog*/metrics_out_unuse_sample > metrics_out_unuse_sample_all
for ii in `seq 100 100 1000`; do grep "^[^,]*,${ii}," metrics_out_unuse_sample_all > metrics_out_unuse_sample_${ii}; done
for ii in `seq 100 100 1000`; do sed -i '1 i\program,iter,wass_result_avg,wass_value_avg,wass_result_ext,wass_value_ext' metrics_out_unuse_sample_${ii}; done

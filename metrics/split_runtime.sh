#!/usr/bin/env bash
#for pp in `cat progs_curr_dir | head `; do cd $pp; echo $pp; /srv/local/scratch/mnt/launch_azure/stanresources/runtime_summary.py output_1000_1.gz > rt_warmup_0601; cd -; done
# for pp in `ls *rt_0601*`; do ppp=${pp%/*}; cat $pp | sed "s/^/${ppp},/"; done > runtime_0601_all
cat *metrics_out* > metrics_all
sed -i '1 i\program,rhat_ref_result_avg,rhat_ref_value_avg,rhat_ref_result_ext,rhat_ref_value_ext,ess_ref_result_avg,ess_ref_value_avg,ess_ref_result_ext,ess_ref_value_ext,rhat_min_result_avg,rhat_min_value_avg,rhat_min_result_ext,rhat_min_value_ext,ess_min_result_avg,ess_min_value_avg,ess_min_result_ext,ess_min_value_ext,tosize,t_result_avg,t_value_avg,t_result_ext,t_value_ext,ks_result_avg,ks_value_avg,ks_result_ext,ks_value_ext,ekl_result_avg,ekl_value_avg,ekl_result_ext,ekl_value_ext,ehell_result_avg,ehell_value_avg,ehell_result_ext,ehell_value_ext,wass_result_avg,wass_value_avg,wass_result_ext,wass_value_ext,js_result_avg,js_value_avg,js_result_ext,js_value_ext' metrics_all
cat *rt_0601* > runtime_0601_all
for ii in `seq 100 100 1000; seq 1010 10 1100`; do grep "^[^,]*,${ii}," runtime_0601_all > runtime_0601_${ii}; done
for ii in `seq 100 100 1000; seq 1010 10 1100`; do sed -i '1 i\program,iter,lp__,accept_stat__,stepsize__,treedepth__, n_leapfrog__,divergent__,energy__' runtime_0601_${ii}; done

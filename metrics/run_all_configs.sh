#!/usr/bin/env bash
# cd ~/Documents/are/c4pp/batchs/timeseries_mutants_7/progs20190621-205419688528
# cat progs_run | shuf | cut -d/ -f1 | xargs -P 3 -n 1 -I {} sh -c "dd={}; cd \$dd; cp ../run* ./; ./run_all_configs.sh .; "

dest_path="\/home\/zixin\/Documents\/are\/c4pp\/batchs\/timeseries_mutants_7_res\/"

    sed -i 's/dest_path=.*$/dest_path='"$dest_path"'/' run_stan_0.config
    sed -i 's/dest_path=.*$/dest_path='"$dest_path"'/' run_stan.config
curr_path=$(realpath .)
curr_path=${curr_path##*/}

dest_path=$(echo $dest_path | sed 's/\\//g')
ls "${dest_path}${curr_path}_res"
# if [ ! -d "${dest_path}${curr_path}_res" ] ; then ~/c4pp/metrics/run_stan_config.sh . ; fi
# 
i=1

# warmup_size=1000
# tree_depth=10
# delta=80
# for sample_size in `seq 200 200 2000`; do
#     sed -i 's/^min=.*$/min='"$sample_size"'/' run_stan_0.config
#     sed -i 's/num_warmup=[0-9]\+/num_warmup='"$warmup_size"'/' run_stan_0.config
#     sed -i 's/delta=[0-9|\.]\+/delta='"0.$delta"'/' run_stan_0.config
#     sed -i 's/-w [0-9]\+/-w '"$warmup_size"'/' run_stan_0.config
#     sed -i 's/_res_[0-9]\+/_res_'"$i"'/' run_stan_0.config
#     sed -i 's/_res_[0-9|_]\+/_res_'"${sample_size}_${warmup_size}_${tree_depth}_${delta}"'/' run_stan_0.config
#     sed -i 's/max_depth=[0-9]\+/max_depth='"$tree_depth"'/' run_stan_0.config
#     ./run_stan_config.sh $1
#     ((i++))
# done

sample_size=1000
tree_depth=10
delta=80
for warmup_size in `seq 200 200 1400` ; do
    sed -i 's/^min=.*$/min='"$sample_size"'/' run_stan_0.config
    sed -i 's/num_warmup=[0-9]\+/num_warmup='"$warmup_size"'/' run_stan_0.config
    sed -i 's/delta=[0-9|\.]\+/delta='"0.$delta"'/' run_stan_0.config
    sed -i 's/-w [0-9]\+/-w '"$warmup_size"'/' run_stan_0.config
    sed -i 's/_res_[0-9]\+/_res_'"$i"'/' run_stan_0.config
    sed -i 's/_res_[0-9|_]\+/_res_'"${sample_size}_${warmup_size}_${tree_depth}_${delta}"'/' run_stan_0.config
    sed -i 's/max_depth=[0-9]\+/max_depth='"$tree_depth"'/' run_stan_0.config
    ./run_stan_config.sh $1
    ((i++))
done

sample_size=1000
warmup_size=1000
delta=80
for tree_depth in 2 4 6 8 10 12 14; do
    sed -i 's/^min=.*$/min='"$sample_size"'/' run_stan_0.config
    sed -i 's/num_warmup=[0-9]\+/num_warmup='"$warmup_size"'/' run_stan_0.config
    sed -i 's/delta=[0-9|\.]\+/delta='"0.$delta"'/' run_stan_0.config
    sed -i 's/-w [0-9]\+/-w '"$warmup_size"'/' run_stan_0.config
    sed -i 's/_res_[0-9|_]\+/_res_'"${sample_size}_${warmup_size}_${tree_depth}_${delta}"'/' run_stan_0.config
    sed -i 's/max_depth=[0-9]\+/max_depth='"$tree_depth"'/' run_stan_0.config
    ./run_stan_config.sh $1
    ((i++))
done
sample_size=1000
warmup_size=1000
tree_depth=10
for delta in 20 40 60 80 90 95 99; do
    sed -i 's/^min=.*$/min='"$sample_size"'/' run_stan_0.config
    sed -i 's/num_warmup=[0-9]\+/num_warmup='"$warmup_size"'/' run_stan_0.config
    sed -i 's/delta=[0-9|\.]\+/delta='"0.$delta"'/' run_stan_0.config
    sed -i 's/-w [0-9]\+/-w '"$warmup_size"'/' run_stan_0.config
    sed -i 's/_res_[0-9|_]\+/_res_'"${sample_size}_${warmup_size}_${tree_depth}_${delta}"'/' run_stan_0.config
    sed -i 's/max_depth=[0-9]\+/max_depth='"$tree_depth"'/' run_stan_0.config
    ./run_stan_config.sh $1
    ((i++))
done
# warmup_size=500
# tree_depth=5
# delta=8
# for sample_size in `seq 100 100 500`; do
#     sed -i 's/^min=.*$/min='"$sample_size"'/' run_stan_0.config
#     sed -i 's/num_warmup=[0-9]\+/num_warmup='"$warmup_size"'/' run_stan_0.config
#     sed -i 's/delta=[0-9|\.]\+/delta='"0.$delta"'/' run_stan_0.config
#     sed -i 's/-w [0-9]\+/-w '"$warmup_size"'/' run_stan_0.config
#     sed -i 's/_res_[0-9|_]\+/_res_'"${sample_size}_${warmup_size}_${tree_depth}_${delta}"'/' run_stan_0.config
#     sed -i 's/max_depth=[0-9]\+/max_depth='"$tree_depth"'/' run_stan_0.config
#     ./run_stan_config.sh $1
#     ((i++))
# done
# tree_depth=5
# delta=8
# for sample_size in `seq 100 100 500`; do
#     warmup_size=$sample_size
#     sed -i 's/^min=.*$/min='"$sample_size"'/' run_stan_0.config
#     sed -i 's/num_warmup=[0-9]\+/num_warmup='"$warmup_size"'/' run_stan_0.config
#     sed -i 's/delta=[0-9|\.]\+/delta='"0.$delta"'/' run_stan_0.config
#     sed -i 's/-w [0-9]\+/-w '"$warmup_size"'/' run_stan_0.config
#     sed -i 's/_res_[0-9|_]\+/_res_'"${sample_size}_${warmup_size}_${tree_depth}_${delta}"'/' run_stan_0.config
#     sed -i 's/max_depth=[0-9]\+/max_depth='"$tree_depth"'/' run_stan_0.config
#     ./run_stan_config.sh $1
#     ((i++))
# done

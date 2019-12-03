#!/usr/bin/env bash
stan_dir=$1
pyro_dir=$2
metrics_file=$3
for results in `find $stan_dir -name "output_100000.gz"`;
do
    d=`echo "$result" | awk -F"/" '{print $(NF-1)}'`
    #echo $results
    echo $d
    pyro_samples=`find $pyro_dir -name "samples" | grep -iw "$d"`

    # if [ `printf "$pyro_samples" | wc -l` -eq 0 ];then
    if [ -z "$pyro_samples" ]; then
	echo "skipping.."
	continue
    else
	echo "using.."
    fi

    if [ -e $pyro_dir/${d}.csv ]; then
	echo "skipping .. result computed.."
	continue
    fi
    

    echo "pyro:  $pyro_samples"
    res=`timeout 10m python $metrics_file -fr $result -fpyro $pyro_samples -m t -m ks -m kl -m smkl -m hell -o avg -o ext -vc` 
    echo "$d,$res"    > $pyro_dir/${d}.csv
done

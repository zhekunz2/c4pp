#!/usr/bin/env bash

# run the stan file for reference iters and min iters 4 times
#     calc all the metrics > ${curr_model_name}_metrics_out_0318_2
#     calc runtime info for 10-100 iters > ${curr_model_name}_rt_0401
#     tar the rest sample files & outputs to the original file path
# Usage:
# ./thisfile stanmodeldirpath
# sed -n '1,1000p' progs_rest | tail |  xargs -P 10 -n 1 -I {} sh -c "./run_em_0401.sh {}"
# (used for running on galeb, where progs_rest contains something like progs20190404-025616567330/progs20190404-025616567330_prob_rand_3
script_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))

source ./run_stan.config
#only run model.stan model.data.R with same source dir name
stan_install_dir=$(realpath $stan_name)
curr_model_name=${input_file_path##*/}
curr_model_name=${curr_model_name}
dest_model_name=${curr_model_name}${dest_model_name_ext}
#if [ ! -f $input_file_path/$curr_model_name.stan ]; then exit 0; fi
if [ `ls $input_file_path/*.stan | wc -l` -eq 0  ]; then echo "no stan file"; echo "no stan file" > $input_file_path/${curr_model_name}_build_error.txt; exit 0; fi
if [ ! -d $stan_install_dir/$curr_model_name ]; then mkdir $stan_install_dir/$curr_model_name; fi
if [ "$sync_all" = true ] || [ ! -f $input_file_path/$curr_model_name.stan ] ; then
    rsync -c $input_file_path/*.data.R $input_file_path/*.stan $stan_install_dir/$curr_model_name
else
    #only copy models with same source file name; cannot be *.data.R *.stan
    rsync -c $input_file_path/$curr_model_name.data.R $input_file_path/$curr_model_name.stan $stan_install_dir/$curr_model_name
fi
cd $stan_install_dir/$curr_model_name
rff=$(realpath .)
echo $rff
cd $stan_install_dir
#only run models with same source file name; cannot be *.stan
if [ -f $rff/$curr_model_name.stan ] ; then
    stanfile=$(ls $rff/$curr_model_name.stan)
else
    stanfile=$(ls $rff/*.stan)
fi
stanfile=${stanfile##*/}
stanfile=${stanfile%.*} # cannot be %%.* if curr_model_name contains .

echo "making..."
# do not recompile

if [ -f $curr_model_name/${stanfile} ] && [ "$recompile" = true ] ; then
    rm -f $curr_model_name/${stanfile}
fi
make $curr_model_name/${stanfile} &> $curr_model_name/build_error.txt
cd $curr_model_name
pwd
if [ ! -f ${stanfile} ]; then echo "$curr_model_name fail to build!"; cp build_error.txt $input_file_path/${curr_model_name}_build_error.txt; exit 0; fi

if [ "$noisy" = true ]; then
    data_file=$(realpath $noisy_data_file)
else
    data_file=$(realpath ${stanfile}.data.R)
fi
if [ `ls $data_file | wc -l` -eq 0  ]; then echo "no data file"; echo "no data file" > $input_file_path/${curr_model_name}_build_error.txt; exit 0; fi

if [  -f csvpipe ]; then rm csvpipe; fi
mkfifo csvpipe

if [ "$variational" = true ] ; then
    vsamples=vb # default
    for i in $(eval echo "{1..$chain}"); do
        timeout ${to}m ./${stanfile} variational algorithm=meanfield data file=${data_file} output file=csvpipe 2>stanerr_${vsamples} | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${vsamples}_$i & $storemethod <csvpipe >output_${vsamples}_${i}${sampleext}
    done
    if [ "$noisy" = true ] ; then
        timeout 8m $stan_install_dir/bin/stansummary output_${vsamples}_*${sampleext} --csv_file=rw_summary_${vsamples}_n &> /dev/null
    else
        timeout 8m $stan_install_dir/bin/stansummary output_${vsamples}_*${sampleext} --csv_file=rw_summary_${vsamples} &> /dev/null
    fi
    cp rw_summary_* $input_file_path
else
    # do sampling, get metrics and rt
if [ "$get_ref" = true ] ; then
    if [ -f ${stanfile}.data.R ]; then
        if [ ! -f ${stanfile}.init.R ]; then
            timeout ${to}m ./${stanfile} sample num_samples=${ref} save_warmup=1 data file=${stanfile}.data.R output file=csvpipe 2>stanerr_${ref} | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${ref} & $storemethod <csvpipe >output_${ref}${sampleext}
        else
            timeout ${to}m ./${stanfile} sample num_samples=${ref} save_warmup=1 data file=${stanfile}.data.R init file=${stanfile}.data.R output file=csvpipe 2>stanerr_${ref} | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${ref} & $storemethod <csvpipe >output_${ref}${sampleext}
        fi
    else
    timeout ${to}m ./${stanfile} sample num_samples=${ref} save_warmup=1 output file=csvpipe 2>stanerr_${ref} | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${ref} & $storemethod <csvpipe >output_${ref}${sampleext}
    fi

    if [ ! -f  $stan_install_dir/$curr_model_name/output_${ref}${sampleext} ] ; then
    # echo "file doesn't exist: $stan_install_dir/$curr_model_name/output_$ref.csv"
        cp stanout_${ref} $dest_path/${curr_model_name}_stanout_${ref}.txt
        cp stanout_${ref} $input_file_path/${curr_model_name}_stanout_${ref}.txt
        exit 0
    fi
    if [ -f rw_summary_${ref} ]; then rm rw_summary_${ref}; fi
    timeout 8m $stan_install_dir/bin/stansummary output_${ref}${sampleext} --csv_file=rw_summary_${ref} &> /dev/null
    if [ ! -f rw_summary_${ref} ];then
        cp stanout_${ref} $dest_path/${curr_model_name}_stanout_${ref}.txt
        cp stanout_${ref} $input_file_path/${curr_model_name}_stanout_${ref}.txt
        exit 0
    else
        cp rw_summary_${ref} $input_file_path/${curr_model_name}_rw_summary_${ref}.txt
    fi
    conv_ref=$($metrics_file_path -c -fs rw_summary_${ref} -m rhat -m ess -o avg -o extreme)
fi

if [ "$get_min" = true ]; then
    rm output_${min}_*

    for i in $(eval echo "{1..$chain}"); do
        timeout ${to}m ./${stanfile} sample num_samples=${min} save_warmup=1 data file=${data_file} output file=csvpipe 2>stanerr_${min}_$i | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${min}_$i & $storemethod <csvpipe >output_${min}_${i}${sampleext}
        # timeout 30m ./${stanfile} variational algorithm=meanfield iter=${min} data file=${data_file} output file=csvpipe 2>stanerr_${min}_$i | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${min}_$i & $storemethod <csvpipe >output_${min}_${i}${sampleext}
        # timeout ${to}m ./${stanfile} variational algorithm=meanfield iter=${min} data file=${data_file} output file=csvpipe 2>stanerr_${min}_$i | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${min}_$i & $storemethod <csvpipe >output_${min}_${i}${sampleext}
    done
    if [ "$noisy" = true ] ; then
        if [ -f rw_summary_${min}_n ]; then rm rw_summary_${min}_n; fi
        timeout 8m $stan_install_dir/bin/stansummary output_${min}_*${sampleext} --csv_file=rw_summary_${min}_n &> /dev/null
        if [ ! -f rw_summary_${min}_n ];then  exit 0; fi
    else
        if [ -f rw_summary_${min} ]; then rm rw_summary_${min}; fi
        timeout 8m $stan_install_dir/bin/stansummary output_${min}_*${sampleext} --csv_file=rw_summary_${min} &> /dev/null
        if [ ! -f rw_summary_${min} ];then  exit 0; fi
    fi
    cp rw_summary_* $input_file_path
    if [ "$get_ref" = true ] ; then
        conv_min=$($metrics_file_path -c -fs rw_summary_${min} -m rhat -m ess -o avg -o extreme)
        #
        # accu=$($metrics_file_path -fr $stan_install_dir/$curr_model_name/output_${ref}${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_1${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_2${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_3${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_4${sampleext} -m t -m ks -m kl -m smkl -m hell -o avg -o extreme -vc)
        accu=$($metrics_file_path -fr $stan_install_dir/$curr_model_name/output_${ref}${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_1${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_2${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_3${sampleext} -fm $stan_install_dir/$curr_model_name/output_${min}_4${sampleext} ${metric_ops})
        #time_1=$(grep " 1 / 101000" stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
        #time_1001=$(grep "1001 / 101000" stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
        #time_2000=$(grep " 2000 / 101000" stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
        #time_e=$(tail -1 stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
        #printf "%s,%s,%s,%s,%s,%s,%s,%s\n" $curr_model_name "$conv_ref" "$conv_min" "$accu" "$time_1" "$time_1001" "$time_2000" "$time_e" | sed 's/ //g' | sed 's/\///g' >> $rff/metrics_out_0317
    fi
    if [ "$get_runtime" = true ] ; then
        $runtime_file_path $stan_install_dir/$curr_model_name/output_${min}_1${sampleext} | sed "s/^/$curr_model_name,/" > rt_0401
    fi
fi
fi

if [ "$archive" = true ] ; then
    mkdir ${dest_model_name}

    if [[ ! -z "$conv_ref" ]] || [[ ! -z "$conv_min" ]] || [[ ! -z "$accu" ]] ; then
        printf "%s,%s,%s,%s\n" $curr_model_name "$conv_ref" "$conv_min" "$accu" | sed 's/ //g' | sed 's/\///g' > ${dest_model_name}/metrics_out_0318_2
        mv ${dest_model_name}/metrics_out_0318_2 $dest_path/${dest_model_name}_metrics_out_0318_2.txt
    fi

    if [ -f rt_0401 ] ; then
        mv rt_0401 ${dest_model_name}/
        mv ${dest_model_name}/rt_0401 $dest_path/${dest_model_name}_rt_0401.txt
    fi
    mv rw_* $dest_model_name
    mv stanout_* $dest_model_name
    mv output*${sampleext} $dest_model_name
    if [ "$tar_archive" = true ]; then
        tar -zcf $dest_model_name.tar.gz $dest_model_name
        mv $dest_model_name.tar.gz $dest_path/
    else
        # mv $dest_model_name $dest_path/
        rsync -av $dest_model_name $dest_path/
        rm -r $dest_model_name
    fi
    rm noisy_*
fi
#for ss in `ls stanout_*`; do mv $ss ${curr_model_name}_$ss; cp ${curr_model_name}_$ss $org_path/${curr_model_name}_$ss.txt; done
#for gg in `ls output_*${sampleext}`; do mv $gg $org_path/${curr_model_name}_$gg; done
#cd $org_path

#mv *${sampleext} /scratch/mnt/em_0401/
#mkdir /scratch/mnt/em_0401/$ff
#mv *${sampleext} /scratch/mnt/em_0401/$ff

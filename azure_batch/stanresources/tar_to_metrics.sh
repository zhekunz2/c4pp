#!/usr/bin/env bash
# ./thisfile input_tar_path.tar.gz

org_path=$(realpath .)
input_file_path=$(realpath $1)
stan_name="../../cmdstan"
ref=100000
min=1000
metrics_file_path=$(realpath ../../metrics_0301.py)
stan_install_dir=$(realpath $stan_name)
curr_model_name=${input_file_path##*/}
curr_model_name=${curr_model_name%%.*}
if [ ! -d $stan_install_dir/$curr_model_name ]; then mkdir $stan_install_dir/$curr_model_name; tar -xf $input_file_path -C $stan_install_dir/$curr_model_name/ --strip-components 1; fi
rm $input_file_path
cd $stan_install_dir/$curr_model_name
rff=$(realpath .)
echo $rff
cd $stan_install_dir
stanfile=$(ls $rff/*.stan)
stanfile=${stanfile##*/}
stanfile=${stanfile%.*}

echo "making..."
#rm -f ${ff}/${ff}
make $curr_model_name/${stanfile}
cd $curr_model_name
pwd
if [ ! -f ${stanfile} ]; then echo "$curr_model_name fail to build!"; exit 0; fi

if [ ! -f csvpipe ]; then rm csvpipe; fi
mkfifo csvpipe
timeout 30m ./${stanfile} sample num_samples=${ref} save_warmup=1 data file=${stanfile}.data.R output file=csvpipe 2>stanerr_${ref} | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${ref} & gzip <csvpipe >output_${ref}.gz
if [ ! -f  $stan_install_dir/$curr_model_name/output_${ref}.gz ] ; then
# echo "file doesn't exist: $stan_install_dir/$curr_model_name/output_$ref.csv"
    exit 0
fi
if [ -f rw_summary_${ref} ]; then rm rw_summary_${ref}; fi
timeout 8m $stan_install_dir/bin/stansummary output_${ref}.gz --csv_file=rw_summary_${ref}
if [ ! -f rw_summary_${ref} ];then exit 0; fi
conv_ref=$($metrics_file_path -c -fs rw_summary_${ref} -m rhat -m ess)

for i in {1..4}; do
    timeout 30m ./${stanfile} sample num_samples=${min} save_warmup=1 data file=${stanfile}.data.R output file=csvpipe 2>stanerr_${min}_$i | perl -pe 'use POSIX strftime; use Time::HiRes gettimeofday; $|=1; select((select(STDERR), $| = 1)[0]); ($s,$ms)=gettimeofday(); $ms=substr(q(000000) . $ms,-6); print strftime "[%s.$ms]", localtime($s)' >stanout_${min}_$i & gzip <csvpipe >output_${min}_$i.gz
done
if [ -f rw_summary_${min} ]; then rm rw_summary_${min}; fi
timeout 8m $stan_install_dir/bin/stansummary output_${min}_*.gz --csv_file=rw_summary_${min}
if [ ! -f rw_summary_${min} ];then  exit 0; fi
conv_min=$($metrics_file_path -c -fs rw_summary_${min} -m rhat -m ess)
#
accu=$($metrics_file_path -fr $stan_install_dir/$curr_model_name/output_${ref}.gz -fm $stan_install_dir/$curr_model_name/output_${min}_1.gz -fm $stan_install_dir/$curr_model_name/output_${min}_2.gz -fm $stan_install_dir/$curr_model_name/output_${min}_3.gz -fm $stan_install_dir/$curr_model_name/output_${min}_4.gz -m t -m ks -m kl -m smkl -m hell)
#time_1=$(grep " 1 / 101000" stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
#time_1001=$(grep "1001 / 101000" stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
#time_2000=$(grep " 2000 / 101000" stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
#time_e=$(tail -1 stanout_100000_$i | cut -d "[" -f2 | cut -d "]" -f1)
#printf "%s,%s,%s,%s,%s,%s,%s,%s\n" $curr_model_name "$conv_ref" "$conv_min" "$accu" "$time_1" "$time_1001" "$time_2000" "$time_e" | sed 's/ //g' | sed 's/\///g' >> $rff/metrics_out_0317
mkdir ${curr_model_name}
printf "%s,%s,%s,%s\n" $curr_model_name "$conv_ref" "$conv_min" "$accu" | sed 's/ //g' | sed 's/\///g' > ${curr_model_name}/metrics_out_0318_2
mv stanout_* $curr_model_name
mv output*.gz $curr_model_name
tar -zcf $curr_model_name.tar.gz $curr_model_name
mv ${curr_model_name}/metrics_out_0318_2 $org_path/${curr_model_name}_metrics_out_0318_2.txt
mv $curr_model_name.tar.gz $org_path/
#for ss in `ls stanout_*`; do mv $ss ${curr_model_name}_$ss; cp ${curr_model_name}_$ss $org_path/${curr_model_name}_$ss.txt; done
#for gg in `ls output_*.gz`; do mv $gg $org_path/${curr_model_name}_$gg; done
cd $org_path

#mkdir /scratch/mnt/lrm_mutant_2/$ff
#mv *.gz /scratch/mnt/lrm_mutant_2/$ff

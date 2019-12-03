#!/usr/bin/env bash
#dir=`realpath $1`
##stanfile=`ls $dir/*.stan`
#name=$1
#mkdir -p ~/projects/c4pp/scripts/cmdstan/$name
#cp $1.data.R $1.stan ~/projects/c4pp/scripts/cmdstan/$name
#cd ~/projects/c4pp/scripts/cmdstan/
##mkdir -p $name
##cp $dir/*.stan $dir/*.data.R $name/
#make $name/$name
#cd $name
#./$name sample num_samples=1 num_warmup=100 data file=${name}.data.R

dir=$1
dirname=`basename $1`
stanfile=`ls $dir/*.stan`
datafile=`ls $dir/*.R`
name=`basename $stanfile | cut -d'.' -f1`
mkdir -p /Users/zhekunz2/Desktop/cmdstan-2.20.0/test_time_series/$dirname
cp $stanfile $datafile /Users/zhekunz2/Desktop/cmdstan-2.20.0/test_time_series/$dirname
cd /Users/zhekunz2/Desktop/cmdstan-2.20.0/
make test_time_series/$dirname/$name &> test_time_series/$dirname/comp
cd test_time_series/$dirname
echo $dirname
./$name sample num_samples=1000 num_warmup=1000 data file=${name}.data.R &> out
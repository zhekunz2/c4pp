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
mkdir -p ~/projects/c4pp/scripts/cmdstan/$dirname
cp $stanfile $datafile ~/projects/c4pp/scripts/cmdstan/$dirname
cd ~/projects/c4pp/scripts/cmdstan/
make $dirname/$name &> $dirname/comp
cd $dirname
echo $dirname
./$name sample num_samples=1000 num_warmup=1000 data file=${name}.data.R &> out
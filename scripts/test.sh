#!/usr/bin/env bash
echo $1
base=`basename $1`
echo $base
#p=`grep -iw $base ~/projects/c4pp/ml/lrm_mutants_7_rhat_adv_metrics.csv | wc -l`
p=1
if [ ! -e $1/ctemplate.template ]; then
    echo "skipping..."    
elif [ -e $1/out.csv ]; then
	echo "skipping csv"
elif [ "$p" -eq 0 ]; then
	echo "no result"
else    
    touch $1/out.csv
    timeout 10m ~/projects/c4pp/feature.py `ls $1/*.stan` res `ls $1/*.data.R` $1/out.csv `ls $1/*.template` #`ls $1/*.py | grep -v "driver.py"`
    #~/projects/c4pp/feature.py `ls $1/*.stan` res dd $1/out_1.csv `ls $1/*.template` `ls $1/*_1.py | grep -v "driver.py"`
fi

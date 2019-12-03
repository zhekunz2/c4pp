#!/usr/bin/env bash
for f in `cat $1 | awk -F"/" '{print "/scratch/lrm_mutant_3/"substr($1, 0 , 26)"/"$1"/ctemplate.template"}'`; do
    echo $f
    ch=`python fix_file.py $f`
    if [ -z $ch ]; then
    echo "skipping..."
    else
    echo $f
    base=`dirname $f`
    datafile=`ls $base/*.data.R`
    printf "\n$ch\n" >> $datafile
    echo "changed..."
    fi
done

#cat $1 | awk -F"/" '{print "../lrm_mutant_3/"substr($1, 0 , 27)"/"$1"/ctemplate.template"}' | xargs -I{} sh -c "echo {}; python ../../scripts/fix_file.py {}"

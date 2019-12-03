#!/usr/bin/env bash
#for k in `find $1 -name "ctemplate.template"`; do
#d=`ls -d */$1`
d=`dirname $1`

cd $d
#f=`find -name "*_1.py" | grep -v "driver"`
f=`basename $1`
echo $f
timeout 30m python $f &> out
cd - > /dev/null
    
#done


#!/usr/bin/env bash
#for k in `find $1 -name "ctemplate.template"`; do
#d=`ls -d */$1`
d=`dirname $1`
tmpname=`echo "$d" | awk -F"/" '{print $(NF)}' `
# in_succeed=`grep "$tmpname" -w succeed_progs -l | wc -l`

#  if [ $in_succeed -eq 0 ] ;then
#     echo "skipping..."
#     exit 1
# fi

cd $d

if [ -e "out" ]; then
    cd - > /dev/null
    echo "done"
    exit 1
fi



#f=`find -name "*_1.py" | grep -v "driver"`
f=`basename $1`
echo $f
timeout 30m python $f &> out
cd - > /dev/null
    
#done


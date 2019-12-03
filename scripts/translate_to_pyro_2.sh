#!/usr/bin/env bash
# Description: this script translates all templates in a directory to pyro programs recursively.. should be parallelized externally
if [ ! -z $2 ]; then
lim=$2
else
lim=1
fi
for f in `find $1 -name "*.stan"`; do
    full=`realpath $f`
    dirname=`dirname $full`
    tmpname=`echo "$dirname" | awk -F"/" '{print $(NF)}' `
    in_succeed=`grep "$tmpname" -w succeed_progs -l | wc -l`

     if [ $in_succeed -eq 0 ] ;then
        echo "skipping..."
        continue
      else
        echo "using..."
    fi

    template=`echo "$dirname/ctemplate.template"`

    if [ ! -e "$template" ]; then
        continue
    fi

    base_template=`basename $template | cut -d "." -f1`
    base_py=`basename $full | cut -d "." -f1`


#    echo $full
#    echo $dirname
#    echo $template
#    echo $base_template
#    echo $base_py
    cp $template /tmp/${tmpname}.template
    echo $tmpname
    dirname=`echo "$dirname" | sed 's/mixture_mutants_1/mixture_mutants_1_pyro/g'`

    mkdir -p $dirname
    for it in `seq 1 $lim`; do
        #(cd ~/projects/Prob-Fuzz/inferno/ && ./test.py /tmp/${tmpname}.template -a SVI -ag -samples -it 4000 -lr 0.001 && mv ${tmpname}.py ${dirname}/${base_py}_${it}.py)
        (cd ~/projects/Prob-Fuzz/inferno/ && ./test.py /tmp/${tmpname}.template -a SVI -ag -samples -it 4000 -lr 0.001 && mv ${tmpname}.py ${dirname}/${base_py}.py)
        cp $template ${dirname}/
    done
    break
done

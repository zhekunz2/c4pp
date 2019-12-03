#!/usr/bin/env bash
for x in `find $1 -name "*.template" | head -2`;
do
    
    b=`dirname $x`
    st=`ls $b/*.stan`
    data=`ls $b/*.data.R`
    #out=`$b/out.csv`;  
    # pyrofile=`ls $b/*.py | grep -v "driver" | grep -v "*_[0-9].py"`
    fname=`echo "$b" | awk -F"/" '{print $(NF)}'`
    #echo $fname
    # mkdir -p programs/lrm_mutants_8/${fname}
    if [[ -e "$b/out_ast2.csv" ]];then
	echo "skipping ... "
	continue
    fi
    # have_stan_results=`grep -i "$fname" $2 | wc -l`
    # if [ $have_stan_results -eq 0 ]; then
    # 	echo "skip, dont have results..."
    # 	continue
    # fi
    echo $b
    #continue
    # touch $b/out_ast.csv
    
    timeout 10m python ./ml/c2v/path.py $x >  "$b/out_ast2.csv"
done


#!/usr/bin/env bash
results=$1
template=$2
iters=`sed  -n -e 's/.*iters=\([0-9]\).*/\1/p'  $1`
programs=`sed  -n -e 's/.*programs=\([0-9]\+\).*/\1/p'  $1`
echo $iters
echo $programs
newfilename=`basename $1`
grep -i  "iteration" $1 > /tmp/tmp_$newfilename

for prog in `seq 1 $programs`; do
    printf "prog_%s," $prog
    for it in `seq 1 $iters`; do
        i=$(($it - 1))
        iter=`grep -i -nr "iteration: $i" /tmp/tmp_$newfilename | sed -n ${prog}p`
        # iter1=`grep -i -nr "iteration: 1" /tmp/tmp_$newfilename | sed -n ${prog}p`
        # iter2=`grep -i -nr "iteration: 2" /tmp/tmp_$newfilename | sed -n ${prog}p`
        transformer=`echo $iter | awk -F"::" '{print $2}'`
        printf "$transformer : "
        if [[ `printf "$iter" | grep "fail" -io | wc -c` -gt 0 ]]; then
                printf "0,"
        else
                printf "1,"
        fi

    done
    echo ""


    continue
    for it in `seq 1 $iters`; do
        i=$(($it - 1))
        iter0=`grep -i -nr "iteration: $i" /tmp/tmp_$newfilename | sed -n ${prog}p`
        iter1
        iterline=`grep -i -nr "iteration: $i" /tmp/tmp_$newfilename | awk -F":" '{print $1}' | sed -n ${prog}p`
        # echo $iterline

        #iterline=`echo "$alliterlines" | sed -n "${prog}p"`
        l=$(($iterline+2))
        iterlines=`sed -n "${iterline},${l}p" /tmp/tmp_$newfilename`
        echo $iterlines
        continue
        transformer=`echo "$iterlines" | grep -i "applying" | cut -d" " -f3`
        skipped=`echo "$iterlines" | grep -i "skip" | wc -l`
        if [ $skipped -eq 0 ]; then
            printf "%s : 1," $transformer
        else
            printf "%s : 0," $transformer
        fi
    done
    printf "\b\n"

done
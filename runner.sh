#!/usr/bin/env bash
export C4PP_ROOT="$HOME/projects/c4pp"

while getopts ":r:f:" opt; do
    case ${opt} in
        r )
            run=$OPTARG
            ;;
        f )
            feature=$OPTARG
            ;;
        : )
            echo "usage: ./runner.sh -r [run?] -f [compute feature?]"
            ;;
    esac
done

programs=`cat programs_list.csv`
touch 'output.csv'

for p in $programs;
do
    folder=`echo $p | awk -F"," '{print $1}'`
    type=`echo $p | awk -F"," '{print $2}'`

    if [ $type == "d" ];then
        results=`cat ./$folder/out`
        for prog in `find ./$folder/* -maxdepth 1 -type d`; do

            filepath=`ls $prog/*.stan`
            #echo $filepath
            ./feature.py $filepath "$folder/out"

        done
    else
        if [ -z $run ] || [ $run == "0" ]; then
            echo $folder
            ./run_progs.sh $folder
        fi
    fi
done
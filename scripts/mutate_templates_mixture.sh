#!/usr/bin/env bash
#seq 1 197 | xargs -n 1 -P 12  -I{} sh -c "./mutate_template.sh lrm.txt 5 {}"
resultdirname=`date +%s`
total=`find ../programs/templates/ -name "*.template" | wc -l`
echo $total
mkdir -p "result_$resultdirname"
seq 1 $total | xargs -n 1 -P 12  -I{} sh -c "./mutate_template.sh mixture_models.txt 600 {} $resultdirname"

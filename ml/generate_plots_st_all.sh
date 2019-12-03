#!/usr/bin/env bash
# convergence plot
run_template (){
class=`echo $2 | cut -d"_" -f1`
if [[ "$3" == "wass" ]]; then
metric_file="$2_metrics_wass.csv"
else
metric_file="$2_metrics.csv"
fi
echo "$metric_file"
./train.py -f csvs/$2_features.csv -fo csvs/$2_features_g2.csv -l csvs/${metric_file} -a rf -m $3 -suf avg -bw -plt -st -saveas plots/${3}_avg_static_${class}_st_${1}.png -tname $1

}

export -f run_template
templates=`cat $1 | grep  -v "\[\|\]\|\{\|\}" | grep -o "[a-zA-Z0-9_-]\+"`
cat $1 | grep  -v "\[\|\]\|\{\|\}" | grep -o "[a-zA-Z0-9_-]\+"  | parallel run_template {} $2 $3


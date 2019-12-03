#!/usr/bin/env bash
dir=$1
basedir=`dirname $0`
#stan
stan_weight=`grep -ir  "se_mean" -A4 $dir/pplout* | tail -4 | awk '{print $2}' | head -1`
#stanbias=`grep -ir  "se_mean" -A4 $dir/pplout* | tail -4 | awk '{print $2}' | head -2 | tail -1`
edward_weight=`cat $dir/edwardout_* | tail -3 | head -1 | grep -io "\-\?[0-9]\+[\.]\?[0-9]\+"`
pyro_weight=`cat $dir/pyroout_* | grep 'w_mean' | awk '{print $2}' | grep -io "\-\?[0-9\.]\+"`

#stan-edward
res=`$basedir/diff.py $stan_weight $edward_weight`

if [[ $res = *"True"* ]]; then
    printf -- '-,'
else
    printf "*,"
fi

#edward-pyro
res=`$basedir/diff.py $edward_weight $pyro_weight`

if [[ $res = *"True"* ]]; then
    printf -- '-,'
else
    printf "*,"
fi

#pyro-stan
res=`$basedir/diff.py $pyro_weight $stan_weight`

if [[ $res = *"True"* ]]; then
    printf -- '-,'
else
    printf "*,"
fi




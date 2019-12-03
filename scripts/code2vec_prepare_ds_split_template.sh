#!/usr/bin/env bash

files=$2
templatename=$3
templatefiles=`cat $files | shuf | parallel  'ls -d  */{}/dogs_log*stan' 2> /dev/null | cut -d"/" -f2`
echo "$templatefiles" > /tmp/tf

filtered_list=`comm -23 <(cat $files | sort) <(echo "$templatefiles" | sort)`
echo "$filtered_list" > /tmp/fl

n=`echo "$filtered_list" | wc -l | cut -d" " -f1`
echo $n

train=`bc <<< "0.8 * $n"`
val=`bc <<< "0.2 * $n + $train"`
#test=`bc <<< "0.1 * $n + $train"`

echo $train
echo $val
i=0

for x in `echo "$filtered_list" | shuf`;
do
p1=`echo $x | cut -d"_" -f1`
path=`echo $1/$p1/$x/ctemplate.template`
#echo $path
if [[ $i < $train ]]; then

mkdir -p $1/train_$3/$p1/$x
cp -r $path $1/train_$3/$p1/$x
#elif [[ $i < $test ]]; then
#mkdir -p $1/test/$p1/$x
#cp -r $path $1/test/$p1/$x/
else
mkdir -p $1/val_$3/$p1/$x
cp -r $path $1/val_$3/$p1/$x
fi
i=$(($i+1))
done

for x in `echo "$templatefiles"`;
do
p1=`echo $x | cut -d"_" -f1`
path=`echo $1/$p1/$x/ctemplate.template`
mkdir -p $1/test_$3/$p1/$x
cp -r $path $1/test_$3/$p1/$x/
done





#!/usr/bin/env bash

files=$2
if [[ ! -z $3 ]]; then
index="_$3"
else
index=""
fi
echo $index

n=`wc -l $files | cut -d" " -f1`
echo $n
train=`bc <<< "0.7 * $n"`
test=`bc <<< "0.2 * $n + $train"`
val=`bc <<< "0.1 * $n + $test"`
echo $train
echo $test
echo $val
i=0

for x in `cat $files | shuf`;
do
p1=`echo $x | cut -d"_" -f1`
path=`echo $1/$p1/$x/ctemplate.template`
#echo $path
if [[ $i < $train ]]; then

mkdir -p $1/train${index}/$p1/$x
cp -r $path $1/train${index}/$p1/$x
elif [[ $i < $test ]]; then
mkdir -p $1/test${index}/$p1/$x
cp -r $path $1/test${index}/$p1/$x/
else
mkdir -p $1/val${index}/$p1/$x
cp -r $path $1/val${index}/$p1/$x
fi
i=$(($i+1))
done



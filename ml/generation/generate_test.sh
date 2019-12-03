#!/usr/bin/env bash
metrics=$1
size=$2
cat $metrics | awk -F"," '$2 == "True" { print $1}' | shuf | head -700 > test1
cat $metrics | awk -F"," '$2 == "False" { print $1}' | shuf | head -300 >> test1

comm -23 <(cat $metrics | awk -F"," '$2 == "True" { print $1}' | sort) <(sort test1) | head -700 > test2
comm -23 <(cat $metrics | awk -F"," '$2 == "False" { print $1}' | sort) <(sort test1) | head -300 >> test2
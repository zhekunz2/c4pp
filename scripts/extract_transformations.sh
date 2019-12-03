#!/usr/bin/env bash
# creates a csv map from results file for transformation
# ./extract_transformations.sh [result_file] [output_file] [metrics file]
filtered=`cat $1  | grep -i "Mutation done\|iteration" | grep -v "fail"`
#printf  "$filtered"
while read i;
do
if [[ "$i" == *"done"* ]];
then it=`echo $i| cut -d" " -f1`;
printf "$it\n";
else trans=`echo $i | cut -d" " -f4`;
printf "$trans,";
fi ;
done < <(printf "$filtered\n") > $2
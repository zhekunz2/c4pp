#!/usr/bin/env bash
scriptfile=$1
file=`grep modelInits $scriptfile | grep -o '(.*)' | sed "s/(\|)\|'//g"`


dname=`dirname $scriptfile`
file="$dname/$file"
echo $file


cat $dname/script.txt | sed 's/samplesSet.*//g' > $dname/script2.txt

if [[ "$file" == *"txt" ]]; then    
    arr=`cat $file | tr "\r\n" " " | sed 's/structure([^)]\+)/0/g'| sed 's/c([^)]\+)/0/g' | grep -o "(.*)" | sed 's/(\|)//g'`
    echo "$arr"
    IFS=","

    for a in $arr;do
	name=`echo $a | awk -F"=" '{print $1}' | grep -o "\S\+"`
	echo $name	
	sed -i "/modelUpdate(10)/a samplesSet(\"$name\")" $dname/script2.txt
    done     
else
    vars=`grep "<-" $file | awk -F"<" '{print $1}'`
    for a in $vars;do
	name=`echo $a | awk -F"=" '{print $1}' | grep -o "\S\+"`
	echo $name	
	sed -i "/modelUpdate(10)/a samplesSet(\"$name\")" $dname/script2.txt
    done     
fi

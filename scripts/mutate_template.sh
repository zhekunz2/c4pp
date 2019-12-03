#!/usr/bin/env bash
# usage ./mutate_templates.sh [model] [no of programs per template] [template #] [result_folder_suffix]
# eg: ./mutate_templates.sh lrm.txt 5 15 1553890963
tt=`mktemp`

files=`(find ../programs/templates/ -name "*.template" | sort -z > $tt) &&  (head -$3 $tt | tail -1)`
rm $tt
echo $files

for template in $files; do
    b=`basename $template | sed 's/.template//g'`
    valid=`cat $1 | grep -v "%" | grep -i "$b" -l | wc -l`

    if [ $valid -eq 0 ]; then
        # skipping files not in current model
        echo "skipping: $template"
        continue
    fi
#    done=`grep -wir "$b" --include="newtemplate.template" ~/projects/StanModels/inferno/output/progs20190324* -l | wc -l`
#    if [ $done -gt 0 ]; then
#        #echo "skipping done...: $template"
#       #continue
#    fi
    echo $template


    fullpath=`realpath $template`
    echo "Running template: $fullpath"
    sleep $((RANDOM % 21))
    (cd /Users/zhekunz2/Desktop/SixthSense/Prob-Fuzz/inferno/ && python2 probfuzz.py $2 -t $fullpath -g -iters 3) &> result_${4}/result_${b}
    echo "Done template: $fullpath"
done

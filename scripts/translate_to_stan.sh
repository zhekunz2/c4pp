#!/usr/bin/env bash
for f in `find $1 -name "*.template"`; do
    full=`realpath $f`
    dirname=`dirname $full`
    #echo $dirname
    stanfile=`find $dirname -name "*.stan" | wc -l`
    if [[ "$stanfile" -eq 0 ]];
    then
            echo $dirname
#            tmpname=`mktemp`
#            cp $full $tmpname
#            tmpbasename=`echo $tmpname | cut -d"/" -f3`
#            (cd ~/projects/Prob-Fuzz/inferno/ && ./teststan.py $tmpname  && mv $tmpbasename.stan $tmpbasename.data.R $dirname/)
    fi

#    echo $full
#    echo $dirname
#    echo $base
#    (cd ~/projects/StanModels/inferno/ && ./test.py $full -a SVI -samples && cp $base.py $dirname/)
done

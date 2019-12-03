#!/usr/bin/env bash
for f in `find programs/templates/ -name "*.template"`; do
    full=`realpath $f`
    dirname=`dirname $full`
    base=`basename $full | cut -d "." -f1`
    echo $full
    echo $dirname
    echo $base
    (cd ~/projects/StanModels/inferno/ && ./test.py $full -a SVI -samples && cp $base.py $dirname/)
done

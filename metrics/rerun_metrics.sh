#!/usr/bin/env bash

ls -d prog*/ | shuf | head -1000 | xargs -n 1 -P10  -I {} sh -c "\
    echo {};
    cd {};
    ../metrics_0301.py -fr output_100000.gz -fm output_1000_1.gz -fm output_1000_2.gz -fm output_1000_3.gz -fm output_1000_4.gz -m t -m ks -m kl -m smkl -m hell -s 250 > metrics_out_250_4
    ../metrics_0301.py -fr output_100000.gz -fm output_1000_1.gz -fm output_1000_2.gz -fm output_1000_3.gz -fm output_1000_4.gz -m t -m ks -m kl -m smkl -m hell -s 500 > metrics_out_500_4
    ../metrics_0301.py -fr output_100000.gz -fm output_1000_1.gz -fm output_1000_2.gz -fm output_1000_3.gz -fm output_1000_4.gz -m t -m ks -m kl -m smkl -m hell -s 750 > metrics_out_750_4
    ../metrics_0301.py -fr output_100000.gz -fm output_1000_1.gz -fm output_1000_2.gz -fm output_1000_3.gz -fm output_1000_4.gz -m t -m ks -m kl -m smkl -m hell -s 1000 > metrics_out_1000_4
    cd -;
    "

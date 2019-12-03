# !/usr/bin/env python
# ls -d ../programs/timeseries_mutants_3/* | awk -F"/" '{print $4}' | parallel -j 9 python compute_mutation_performance.py {} ../ml/timeseries_mutants_3_metrics_0401.csv kl_result_avg /home/saikat/projects/c4pp/scripts/mutation_scores_ts_3
#ls -d ../programs/timeseries_mutants_7/* | parallel python compute_mutation_performance.py {} ../ml/csvs/timeseries_mutants_7_metrics.csv wass_result_avg result_timeseries_mutants_7 mutation_scores_ts_7
import sys
import subprocess as sp
import re
import pandas as pd
import numpy as np
import os

def update_score(scores, mut, result , score=1.0):
    if mut not in scores:
        scores[mut] = [0, 0]
    if result:
        scores[mut][0]+=score
    else:
        scores[mut][1]+=score


# parse args
template_folder=sys.argv[1]
metrics_file=pd.read_csv(sys.argv[2], index_col='program').fillna(0).replace('inf', np.inf).replace('-inf', np.inf)
metric = sys.argv[3]
scores = dict()
results_folder = sys.argv[4]
template_name = open("{0}/newtemplate.template".format(template_folder)).read().split("/")[-2]
results_file = "{0}/result_{1}".format(results_folder, template_name)
results, err = sp.Popen(["./map_mutations2.sh " + results_file], stdout=sp.PIPE, shell=True).communicate()
#metric='kl' #13
#column=13
#column = metrics_file[0].split(',').index('{0}'.format(metric))
#print(column)
#print(results)

print(template_name)
mutant_folder = template_folder.split('/')[-1]
indices = metrics_file.index.tolist()
#exit(0)
for i in range(1, 601):
    f="{0}_prob_rand_{1}".format(mutant_folder, i)
    mutators = filter(lambda x: 'prog_{0},'.format(i) in x, results.split('\n'))
    if not f in indices:
        continue

    if f in indices:
        metric_val = metrics_file.ix[f][metric]

    if len(mutators) == 1:
        total_mutators = len(re.findall(': 1', mutators[0]))
        for m in mutators[0].split(',')[1:-1]:
            if ': 1' in m:
                if 'True' in str(metric_val):
                    update_score(scores, m.split(':')[0].strip(), True, score=(1.0/total_mutators))
                else:
                    update_score(scores, m.split(':')[0].strip(), True, score=-(1.0/total_mutators))
    else:
        print("skipping " + str(i))

print(template_folder)
print(map(lambda x : (x, scores[x]),sorted(scores.keys(), key=lambda x: scores[x][0], reverse=False)))
if not os.path.exists(sys.argv[5]):
    os.makedirs(sys.argv[5])

with open(sys.argv[5]+'/'+template_name, 'w') as f:
    f.write(str(scores))
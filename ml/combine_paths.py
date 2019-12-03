#!/usr/bin/env python
import ast
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# file1=ast.literal_eval(open(sys.argv[1]).read())
# file2=ast.literal_eval(open(sys.argv[2]).read())
xlabels = {'kl': 'Threshold of KL Divergence',
           'klfix' : 'Threshold of KL Divergence',
           'rhat_min': 'Threshold for Gelman-Rubin Diagnostic',
           'wass' : 'Threshold of Wasserstein Dist'}

def aggregate(f, arr, groups, field):
    if f == np.std:
        return [[np.mean([x[field] for x in part]) - np.min([x[field] for x in part])
                for part in np.split(np.array(arr), groups)],
                [np.max([x[field] for x in part]) - np.mean([x[field] for x in part])
                 for part in np.split(np.array(arr), groups)]]

    else:
        return [f([x[field] for x in part]) for part in np.split(np.array(arr), groups)]


def plot_results(metric_label, exp_results, metric_thresholds, ptype):
    font = {'family': 'normal',
            'size': 20}
    linestyle ={"normal" : "-", "maj": "--"}
    measures = {"F1": {"normal" : "F1", "maj": "F1-maj"}, "AUC": {"normal" : "AUC", "maj": "AUC-maj"}}
    colors={"F1": "C0", "AUC": "C1"}
    matplotlib.rc('font', **font)
    ax = plt.gca()

    from matplotlib.ticker import FormatStrFormatter
    if metric_label in xlabels and xlabels[metric_label] == 'MCMC Iterations':
        pass
    elif metric_label == 't' or metric_label == 'ks':
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for measure in measures.keys():
        plt.errorbar(metric_thresholds, aggregate(np.mean, exp_results, len(metric_thresholds), measure) , label=measures[measure][ptype],
                     marker='s', linewidth=3.0, yerr=aggregate(np.std, exp_results, len(metric_thresholds), measure), color=colors[measure],

                     linestyle=linestyle[ptype])

    plt.xticks(metric_thresholds)
    plt.grid(True)
    plt.ylim((0.45, 1.05))
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if metric_label in xlabels:
        plt.xlabel(xlabels[metric_label])
    plt.ylabel("Scores")
    # plt.legend()
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.11), ncol=4, prop={'size': 14}, columnspacing=0.3, handlelength=3)
    plt.tight_layout()
    # annotate(threshold, [x["Precision"] for x in results])


f1=[]
for size in [2,5,10,20]:
    f1_local = []
    for model in ['lrm', 'timeseries', 'mixture']:
        data=ast.literal_eval(open('results/{0}_avg_static_{1}_ast_{2}.png.txt'.format(sys.argv[1],model, size)).read())

        f1_local.append(aggregate(np.mean, data["results"], len(data["thresholds"]), "F1")[0 if sys.argv[1] == 'rhat' else 1])
    f1.append(np.mean(f1_local))
font = {'family': 'normal',
            'size': 20}
matplotlib.rc('font', **font)
plt.bar([1,2,3,4], f1, color='C2')
plt.ylim(0.8,1.0)

plt.xticks([1,2,3,4], [2,5,10,20])
plt.xlabel("Motif Size")
plt.ylabel("Avg. F1 Scores")
plt.tight_layout()
plt.show()
print(f1)
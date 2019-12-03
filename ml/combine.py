#!/usr/bin/env python
import ast
import sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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
    linestyle ={"normal" : "-", "maj": "--", "c2v": "dotted"}
    measures = {"F1": {"normal" : "F1", "maj": "F1-maj", "c2v": "F1-c2v"}, "AUC": {"normal" : "AUC", "maj": "AUC-maj", "c2v" : "AUC-c2v"}}
    colors={"F1": "C0", "AUC": "C1"}
    markers = {"normal" : "s", "maj": "X", "c2v": "D"}
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
        if measure in exp_results[0]:
            if ptype == 'c2v':
                print([x[measure] for x in exp_results])
            plt.errorbar(metric_thresholds,
                         aggregate(np.mean, exp_results, len(metric_thresholds), measure) if ptype != 'c2v' else
                         [x[measure] for x in exp_results],
                         label=measures[measure][ptype],
                         marker=markers[ptype],
                         linewidth=3.0,
                         yerr=aggregate(np.std, exp_results, len(metric_thresholds), measure) if ptype != 'c2v' else None,
                         color=colors[measure],
                         linestyle=linestyle[ptype])

    plt.xticks(metric_thresholds)
    plt.grid(True)
    plt.ylim((0.45, 1.05))
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if metric_label in xlabels:
        plt.xlabel(xlabels[metric_label])
    plt.ylabel("Scores")
    #plt.legend()
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.11), ncol=3, prop={'size': 13}, columnspacing=0.2, handlelength=3)
    plt.tight_layout()
    # annotate(threshold, [x["Precision"] for x in results])


def plot_bar(c2v_file, c2s, ss, metric):
    font = {'family': 'normal',
            'size': 20}
    matplotlib.rc('font', **font)
    print(x_indices)
    width=0.35
    c2v = list(filter(lambda x: x.split(',')[0] == str(metric_val), open(c2v_file).readlines()))[0]
    c2s = list(filter(lambda x: x.split(',')[0] == str(metric_val), open(c2s).readlines()))[0]
    #c2v = [x.split(',')[c2v_map[metric]] for x in open(c2v_file).readlines()]
    #c2v_f1 = float(c2v[0] if sys.argv[4] == 'rhat_min' else c2v_results_f1[1])
    c2v_f1 = float(c2v.split(',')[c2v_map[metric]])
    c2s_f1 = float(c2s.split(',')[c2s_map[metric]]) if metric in c2s_map else 0.0
    ss_f1 = float([x.split(',')[c2v_map[metric]] for x in open(ss).readlines()][0])
    print(metric, c2v_f1, c2s_f1, ss_f1)
    #ss_f1 = aggregate(np.mean, ss["results"], len(ss["thresholds"]), metric)[0]
    plt.bar([x_indices[c2v_map[metric] - 1] - width / 2], [c2v_f1], width=width / 3,  color="C0")
    plt.bar([x_indices[c2v_map[metric] - 1]], [c2s_f1], width=width / 3,  color="C1")
    plt.bar([x_indices[c2v_map[metric] - 1] + width / 2], [ss_f1], width=width / 3,  color="C2")
    #plt.bar([x_indices[c2v_map[metric]-1] - width/2 , x_indices[c2v_map[metric]-1], x_indices[c2v_map[metric]-1] + width/2], [c2v_f1, c2s_f1, ss_f1], width=width/3, label=["c2v", "c2s", "ss"])



if __name__ == '__main__':
    if 'ex' in sys.argv[1]:
        sixthsense_results = sys.argv[1]
    else:
        sixthsense_results=ast.literal_eval(open(sys.argv[1]).read())

    maj_results=ast.literal_eval(open(sys.argv[2]).read())
    c2v_results_f1=[x.split(',')[1] for x in open(sys.argv[3]).readlines()]
    c2v_results_auc=[x.split(',')[2] for x in open(sys.argv[3]).readlines()]
    print(sys.argv[3])
    #print(file1["results"])
    #print(file2["results"])
    #exit(1)

    # plot_results(sys.argv[4], sixthsense_results["results"], sixthsense_results["thresholds"], "normal")
    # plot_results(sys.argv[4], maj_results["results"], maj_results["thresholds"], "maj")
    # # code2vec
    # plot_results(sys.argv[4], [{'F1': float(x.strip())} for x in c2v_results_f1], maj_results["thresholds"], "c2v")
    # plot_results(sys.argv[4], [{'AUC': float(x.strip())} for x in c2v_results_auc], maj_results["thresholds"], "c2v")
    # if len(sys.argv) > 5:
    #     plt.savefig(sys.argv[5])
    # else:
    #     plt.show()

    # compare sixthsense, codevec, code2seq
    # c2s format is F1, Precision, Recall

    c2s_results=open(sys.argv[3].replace("c2v/new", "c2s")).read().strip().split(",")
    c2v_map = {"F1": 1, "AUC": 2, "Precision": 3, "Recall": 4}
    c2s_map = {"F1": 1, "Precision": 2, "Recall": 3}
    metric_val = 0.05 if sys.argv[4] == 'wass' else 1.05
    import matplotlib.patches as mpatches



    x_indices = np.arange(len(c2v_map.keys()))
    fig, ax = plt.subplots()
    ax.set_xticks(x_indices)
    ax.set_xticklabels(["F1", "AUC", "Prec", "Rec"])
    #plt.ylim((0.7, 1.0))
    plot_bar(sys.argv[3], sys.argv[3].replace("c2v/new", "c2s"), sixthsense_results, "F1")
    plot_bar(sys.argv[3], sys.argv[3].replace("c2v/new", "c2s"), sixthsense_results, "Precision")
    plot_bar(sys.argv[3], sys.argv[3].replace("c2v/new", "c2s"), sixthsense_results, "Recall")
    plot_bar(sys.argv[3], sys.argv[3].replace("c2v/new", "c2s"), sixthsense_results, "AUC")
    plt.legend(handles=[mpatches.Patch(color='C0', label='c2v'), mpatches.Patch(color='C1', label='c2s'),
                        mpatches.Patch(color='C2', label='ss')], bbox_to_anchor=(0,1.04), loc='lower left', ncol=3)
    plt.tight_layout()
    #plt.legend()
    if len(sys.argv) > 5:
        plt.savefig(sys.argv[5].replace(".png", "_all.png"))
    else:
        plt.show()

#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import matplotlib

def read_all_csvs(csv_files, index=None):
    if csv_files is None:
        print("No files")
        exit(-1)
    full_data = None
    for file in csv_files:
        if full_data is None:
            full_data = pd.read_csv(file, index_col=index)
        else:
            df = pd.read_csv(file, index_col=index)
            full_data=full_data.append(df)

    return full_data


def plot(table, metric, template=None):
    font = {'family': 'normal',
            'size': 28}

    matplotlib.rc('font', **font)
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    values = table.fillna(0).replace(np.inf, 9999999).replace(-np.inf, -9999999)[metric]

    if template is not None:
        ind = filter(lambda x: sum([i in x for i in template]) > 0, values.index.tolist())
        print(len(ind))
        values=values.ix[ind]
    #values=values.dropna()
    if 'rhat' in metric:
        values = values.apply(lambda x: 10 if x > 10 else x)
        logbins = np.geomspace(max(np.min(values), 0.8), np.max(values), 20)
        plt.xlabel('Gelman-Rubin Diagnostic')
    else:
        ax.set_xscale('log')
        logbins = np.geomspace(max(np.min(values), 10 ** (-5)), 1000, 30)
        plt.xlabel('KL Divergence')
        plt.ylabel('Mutants')
        # xlabels = np.geomspace(max(np.min(values), 10 ** (-5)), 10000, 5).astype(str)
        # xlabels[-1] += '+'
        # ax.set_xticklabels(xlabels)

    plt.hist(np.clip(values,logbins[0], logbins[-1]), bins=logbins)

    plt.show()


if __name__ == '__main__':
    df = read_all_csvs(sys.argv[1:-1], index='program')
    # pd.read_csv(sys.argv[1], index_col='program').astype(np.float32)
    plot(df, sys.argv[-1], template=['progs20190330-174717050219', 'progs20190329-135448180434'])
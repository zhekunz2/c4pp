#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

def plot_scat(met, name="all"):
    for nn in list(met):
        if "results" in nn:
            met[nn] = met.apply(lambda x: np.nan if pd.isna(x[nn.replace("results", "value")]) else x[nn], axis=1)
    todrop=[mm for mm in list(met) if "results" in mm]
    todrop.append("template")
    met.drop(todrop, axis=1, inplace=True, errors="ignore")
    fig = plt.figure(figsize=(15,3))
    fig.subplots_adjust(hspace=0.4,wspace=0.4)
    for idx, mm in enumerate([xx for xx in list(met) if ("prog" not in xx and "iter" not in xx)]):
        if "hell" in mm:
            met[mm] = met[mm].apply(float).replace([np.inf, -np.inf], 100).dropna().apply(lambda x: 1.2 if x > 1 else x)
        else:
            met[mm] = met[mm].apply(float).replace([np.inf, -np.inf], 100).dropna().apply(lambda x: 10 if x > 10 else x)
        ax = fig.add_subplot(1,len(list(met))-2,idx+1)
        groups = met.groupby('program')
        for idx, (name, group) in enumerate(groups):
            #group.plot(x='iter',y=mm, marker='o', kind='scatter', ax=ax)
            if idx % 2 == 0:
                ax.plot(group.iter,group[mm], marker='o',ms=2)
        #if "kl" in mm or "hell" in mm or "rhat" in mm:
        #    if "rhat_min" in mm:
        #        mm_toplot = met[mm].apply(float).dropna().apply(lambda x: 10 if x > 10 else x)
        #        logbins = np.geomspace(max(mm_toplot.min(),0.8), mm_toplot.max(), 15)
        #    else:
        #        mm_toplot = met[mm].apply(float).replace([np.inf, -np.inf], 100).dropna()
        #        logbins = np.geomspace(max(mm_toplot.min(),10**(-4)), mm_toplot.max(), 15)
        #    mm_toplot.plot.hist(ax=ax, bins=logbins)
        #    ax.set_xscale('log')
        #else:
        #    mm_toplot = met[mm].apply(float).replace([np.inf, -np.inf], 100).dropna()
        #    mm_toplot.plot(ax=ax, bins=15)
        ax.title.set_text(mm)
        if "kl" in mm or "hell" in mm or "rhat" in mm:
            ax.set_yscale('log')
    #fig.suptitle("Hist of " + name)
    plt.show(fig)
    #fig.savefig("../hist_pngs/" + name + "_log_iter.png" , bbox_inches='tight')


met=pd.read_csv(sys.argv[1])
if len(sys.argv) == 2:
    plot_scat(met)
elif "t" in sys.argv[2]:
    met.dropna(subset=["program"], inplace=True)
    #met["template"]=met["program"].apply(lambda x: print(x) if not isinstance(x, str) else x) #x[:26])
    met["template"]=met["program"].apply(lambda x: x[:26])
    temp_mets=met.groupby("template")
    for idx, (name, mm) in enumerate(temp_mets):
        if idx % 7 == 0:
            plot_hist(mm.copy(), name)
        if idx >= 150:
            break


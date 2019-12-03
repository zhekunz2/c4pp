#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd 

met=pd.read_csv(sys.argv[1])
for nn in list(met):
    if "results" in nn:
        met[nn] = met.apply(lambda x: np.nan if pd.isnull(x[nn.replace("results", "value")]) else x[nn], axis=1)
        print(nn)
todrop=[mm for mm in list(met) if "results" in mm]
met.drop(todrop, axis=1, inplace=True)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4,wspace=0.4)
for idx, mm in enumerate(list(met)):
    print(idx)
    print(mm)
    if mm == 'program':
        continue
    ax = fig.add_subplot(1,5,idx)
    ax.set_xlabel(mm)
    met[mm].apply(float).dropna().plot.hist(ax=ax, bins=15)
plt.show(fig)

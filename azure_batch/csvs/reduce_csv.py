#!/usr/bin/env python


import pandas as pd
import sys
import os

output_file = "reduce.csv"
try:
    path = sys.argv[1]
except:
    path = "./"

dfs = []

for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".csv")) and "reduce" not in name and "metric" not in name and "mnt" not in name:
            print(name)
            realfile = os.path.join(root, name)
            dfs.append(pd.read_csv(realfile))
df0 = dfs[0]
for idx in range(1, len(dfs)):
    df0 = df0.append(dfs[idx])

#df0 = df0.drop_duplicates(subset='program')
#df0 = df0[df0.results != "Error"]
namelist = list(df0)
namelist.remove('program')
#namelist.remove('value')
order = ['program']#,'value']
order.extend(namelist)
df0[order].to_csv(output_file, index=False)

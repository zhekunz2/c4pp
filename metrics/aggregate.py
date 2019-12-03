#!/usr/bin/env python3
#./thisfile metricsfile_for_ref_min

import pandas as pd
import numpy as np
import sys
met=pd.read_csv(sys.argv[1])
met["template"]=met["program"].apply(lambda x: x[:26])
for nn in list(met):
    if "results" in nn:
        met[nn] = met.apply(lambda x: np.nan if pd.isna(x[nn.replace("results", "value")]) else x[nn], axis=1)
        met[nn.replace("results", "count")] = met.apply(lambda x: 0 if pd.isna(x[nn.replace("results", "value")]) else 1, axis=1)
# met.dropna(inplace=True)
met["count"] = 1
met["ks_result"] = met["ks_result"].apply((lambda x: True if x == "True" else False))
agg=met.groupby("template").agg({
    'count':'sum',
    'rhat_ref_result':'sum',
    'ess_ref_result':'sum',
    'rhat_min_result':'sum',
    'ess_min_result':'sum',
    't_result':'sum',
    'ks_result':'sum',
    'kl_result':'sum',
    'smkl_result':'sum',
    'hell_result':'sum',
    'rhat_ref_count':'sum',
    'ess_ref_count':'sum',
    'rhat_min_count':'sum',
    'ess_min_count':'sum',
    't_count':'sum',
    'ks_count':'sum',
    'kl_count':'sum',
    'smkl_count':'sum',
    'hell_count':'sum'
    })
agg = agg.replace(False,0)
agg = agg.replace(True,1)
#agg2 = agg[agg["count"] != 1]
agg2 = agg.copy()
for nn in list(agg):
    if "results" in nn:
        agg[nn] = agg.apply(lambda x: "T" if x[nn.replace("results", "count")] == x[nn] else "F" if x[nn] == 0 else "TF", axis=1)
        agg2[nn] = agg2.apply(lambda x: "1" if x[nn.replace("results", "count")] == 1 else "T" if x[nn.replace("results", "count")] == x[nn] else "F" if x[nn] == 0 else "TF", axis=1)
count_cols = [cc for cc in list(agg) if "count" in cc ]
pd.options.display.float_format = '{:,.0f}'.format
print(agg.drop(count_cols,axis=1).apply(pd.value_counts).fillna(0))
print(agg2.drop(count_cols,axis=1).apply(pd.value_counts).fillna(0))

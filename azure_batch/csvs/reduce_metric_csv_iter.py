#!/usr/bin/env python


import pandas as pd
import sys
import os
import numpy as np

try:
    path = sys.argv[1]
except:
    path = "./"

output_file = os.path.join(path, "reduce_metric.csv")
df_base = pd.DataFrame(columns=['program', 'iter_100_nuts_result', 'iter_600_nuts_result', 'iter_1100_nuts_result', 'iter_1600_nuts_result', 'iter_2100_nuts_result', 'iter_2600_nuts_result', 'iter_3100_nuts_result', 'iter_3600_nuts_result', 'iter_4100_nuts_result', 'iter_4600_nuts_result', 'iter_5100_nuts_result'])
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".csv")) and "reduce" not in name and "mnt" not in name and "feature" not in name:
            iter_num = [nn for nn in name.replace('.','_').split('_') if nn.isdigit() and int(nn) % 100 == 0 and int(nn) != 0]
            if len(iter_num) < 1:
                continue
            realfile = os.path.join(root, name)
            df_curr = pd.read_csv(realfile, names = ["program", "parameter", "t_result", "t_value", "ks_result", "ks_value", "kl_result", "kl_value", "smkl_result", "smkl_value", "hell_result", "hell_value", "rhat1_result", "rhat1_value", "rhat2_result", "rhat2_value"], dtype={'rhat1_result':str, 'rhat2_result':str})
            #df_curr.dropna(how='all', inplace=True)
            #df_curr = df_curr.apply(lambda x: True if x == "" else x)

            if "_vb_" in name:
                df_curr = df_curr.drop(['parameter', 'rhat1_result', 'rhat1_value', 'rhat2_value', "t_value", "t_result", "ks_value", "ks_result", "kl_value", "kl_result", "smkl_value", "smkl_result", "hell_value", "hell_result",  "value", "results"], axis=1, errors='ignore')
                #df_curr["rhat2_result"] = df_curr["rhat2_result"].fillna(value=False)
                #df_curr[results] = df_curr[results].astype('bool')
                #df_curr["rhat2_result"] = df_curr["rhat2_result"].astype(int)
                df_curr.rename(columns={'rhat2_result': "iter_" + iter_num[0] + '_nuts_result'}, inplace=True)

            elif "_hmc_" in name:
                df_curr = df_curr.drop(['parameter', 'rhat2_result', 'rhat2_value', 'rhat1_value', "t_value", "t_result", "ks_value", "ks_result", "kl_value", "kl_result", "smkl_value", "smkl_result", "hell_value", "hell_result",  "value", "results"], axis=1, errors='ignore')
                #df_curr["rhat1_result"] = df_curr["rhat1_result"].fillna(value=False)
                #df_curr["rhat1_result"] = df_curr["rhat1_result"].astype(int)
                df_curr.rename(columns={'rhat1_result': "iter_" + iter_num[0] + '_hmc_result'}, inplace=True)
            else:
                continue
            #if df_base.empty:
            #    df_base = df_curr
            #else:
            #df_base.set_index('program')
            df_dupl = pd.DataFrame(columns=['program'])
            for index, row in df_curr.iterrows():
                if df_base.empty or row['program'] not in list(df_base.program):
                    df_base = df_base.append(row)
                    if "seeds_centered" in row['program']:
                        print("not in {}".format(row['program']))

                else:
                    if "seeds_centered" in row['program']:
                        print(row['program'])
                        print("in {}".format(row['program']))
                    #    df_dupl = df_dupl.append(row)
                    #    df_base.update(df_dupl, overwrite=True, filter_func=lambda x: True)
                    #    df_base.to_csv(output_file, index=False)
                    #    exit(0)
                    df_dupl = df_dupl.append(row)
            df_base.update(df_dupl, overwrite=True, filter_func=lambda x: pd.isna(x))

#df0 = df0.drop_duplicates(subset=['program','iter'])
#df0 = df0[df0.results != "Error"]
#namelist = list(df_base)
#namelist.remove('program')
#namelist.remove('value')
#order = ['program']#,'value']
order = ['program', 'iter_100_nuts_result', 'iter_600_nuts_result', 'iter_1100_nuts_result', 'iter_1600_nuts_result', 'iter_2100_nuts_result', 'iter_2600_nuts_result', 'iter_3100_nuts_result', 'iter_3600_nuts_result', 'iter_4100_nuts_result', 'iter_4600_nuts_result', 'iter_5100_nuts_result']
#order.extend(namelist)
df_base[order].to_csv(output_file, index=False)
#df_base.to_csv(output_file)#, index=False)

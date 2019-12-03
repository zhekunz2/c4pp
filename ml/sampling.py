import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1], index_col='program')
df=df.fillna(0)
bins=np.histogram(df.wass_value_avg, range=(0.0,3.0), bins=15)
digs=np.digitize(df.wass_value_avg, bins[1])
samples=[]
bin_choices = [np.random.choice(range(1,len(bins[1])+1)) for _ in range(10000)]
indices = [filter(lambda x: digs[x] == bin, range(len(df))) for bin in bin_choices]
[samples.append(list(df.wass_value_avg)[np.random.choice(ind)]) for ind in indices]
# for i in range(10000):
#     bin=np.random.choice(range(1,len(bins[1])+1))
#     #print(bin)
#     indices=filter(lambda x: digs[x] == bin, range(len(df)))
#     samples.append(list(df.wass_value_avg)[np.random.choice(indices)])
#print(samples)
plt.hist(samples, bins=bins[1])
plt.show()
plt.hist(df.wass_value_avg, bins=bins[1])
plt.show()

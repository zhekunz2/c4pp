#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

def update(d, val):
    if val in d:
        d[val] += 1
    else:
        d[val] = 0


rbp = RandomBinaryProjections('rbp', 5)
rbp.reset(1)
#engine = Engine(10, lshashes=[rbp])


file = open(sys.argv[1]).readlines()

df = pd.DataFrame()
bigdict=dict()
i=0
for line in file:
    data = dict()
    triples=line.split(" ")[1:]

    for t in triples:
        update(data, rbp.hash_vector([float(t.split(",")[1])])[0])
    # for p in range(0, 200-len(data.values())):
    #     data[p]=0
    # for bucket_key in rbp.hash_vector([float(x) for x in data.values()]):
    #     print(bucket_key)
    #     bigdict[i] = {'key' : bucket_key}
    #df=df.append(data, ignore_index=True)
    bigdict[i]=data
    i=i+1

df = pd.DataFrame(bigdict)
print(len(bigdict.keys()))
df=df.transpose().fillna(0)

df.to_csv('test.csv')
print(df.head(2))
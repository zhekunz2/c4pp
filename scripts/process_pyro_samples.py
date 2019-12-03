#!/usr/bin/env python
import sys
import numpy as np
import io
import re
import ast
file=open(sys.argv[1]).read().splitlines()
data = {}
for f in file:
    name=f.split(':')[0]
    print(name)
    samples=f.split(':')[1]
    cur_arr = np.array(ast.literal_eval(samples.replace('nan', '\"nan\"')))
    # if "[[[" in samples:
    #     arrs = re.findall("\[[^\[\]]+\]", samples)
    #     cur_arr=np.array(ast.literal_eval(samples))
    #     print(cur_arr.shape)
    #
    # elif "[[" in samples:
    #     #1d or scalar
    #     arrs=re.findall("\[[^\[\]]+\]", samples)
    #     print(len(arrs))
    #     for arr in arrs:
    #         a=np.fromstring(arr.replace('[', '').replace(']', ''),  sep=',')
    #         #print(a)
    #         if a.size == 1:
    #             # scalar variable
    #             cur_arr = np.append(cur_arr, a)
    #         else:
    #             if len(cur_arr) == 0:
    #                 cur_arr = a
    #             else:
    #                 cur_arr = np.row_stack((cur_arr,a))

    print(cur_arr.shape)
    data[name]=cur_arr

def getSamples(name, indices):
    if indices is None:
        samples =data[name]
    elif type(indices) == np.int or len(indices) == 1:
        d = data[name]
        samples = [x[indices] for x in d]
    elif len(indices) == 2:
        d = data[name]
        samples = [x[indices[0]][indices[1]] for x in d]
    else:
        samples = []
    return samples

print(getSamples('sigma', None))

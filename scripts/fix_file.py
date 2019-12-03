#!/usr/bin/env python
import sys
import re
import ast
import numpy as np
tmp=open(sys.argv[1]).read()
data=re.findall(r'^[a-zA-Z0-9_]+\s*:[^=\n]+', tmp, flags=re.MULTILINE)
indices=dict()
arrs=dict()
for d in data:
    ind=re.search("[\[]([a-zA-Z_0-9]+[a-zA-Z0-9_]*)(,[a-zA-Z_0-9]+[a-zA-Z0-9_]*)*[\]]", d)

    if ind is not None:
        #print(ind.group(1))
        arr = re.search("[\[<][0-9\.,]+[>\]]", d.replace(" ", ""))

        if arr is not None:
            #print("arr: " + str(arr.group(0)))
            nparr=ast.literal_eval(arr.group(0).replace("<", "[").replace(">", "]"))
            size=len(nparr)
            arrs[ind.group(1)] = size
            #print(size)
    else:
        #print(d)
        try:
            indices[d.split(":")[0].strip()] = int(d.split(":")[1].strip())
        except Exception as e:
            print(e)

#print(indices)
#print(arrs)
for a in arrs:
    if a in indices and indices[a] != arrs[a]:
        print("{0} <- {1}".format(a, arrs[a]))

#print(data)
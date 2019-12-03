#!/usr/bin/env python
from combine import *

file1=ast.literal_eval(open(sys.argv[1]).read())
name=sys.argv[1].split("timeseries")[1].replace(".png.txt", "")
print('{0}&{1}&{2}\\\\'.format(name.replace("_","\_"),
    np.round(aggregate(np.mean, file1["results"], len(file1["thresholds"]), "F1")[1],2),
      np.round(aggregate(np.mean, file1["results"], len(file1["thresholds"]), "AUC")[1], 2)))


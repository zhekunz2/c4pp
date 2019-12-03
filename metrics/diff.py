#!/usr/bin/env python
import os
import json
import sys
import numpy as np

try:
    op1=float(sys.argv[1])
    op2=float(sys.argv[2])
except:
    print('False : Err')
    exit(0)

smape = abs(op1 - op2) / ((abs(op1) + abs(op2)))

if abs(smape) <= 0.1:
    print('True: ' + str(smape))
else:
    print('False: ' + str(smape))

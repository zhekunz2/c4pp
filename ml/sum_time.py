#!/usr/bin/env python

import pandas as pd
import sys
import time as tm

tfpn = pd.read_csv(sys.argv[1])
tfpn.set_index("program", inplace=True)
time = pd.read_csv(sys.argv[2])
time.set_index("program", inplace=True)
joint = tfpn.join(time)
print("Total: {}".format(len(joint.index)))
join_TP=joint[joint["TFPN"] == "TP"]
join_FP=joint[joint["TFPN"] == "FP"]
join_TN=joint[joint["TFPN"] == "TN"]
join_FN=joint[joint["TFPN"] == "FN"]
print("TP: {}+{}={}".format(join_TP["time"].sum()*4,len(join_TP.index),join_TP["time"].sum()*4+len(join_TP.index)*41.0309))
print("FP: {}+{}={}".format(join_FP["time"].sum()*4,len(join_FP.index),join_FP["time"].sum()*4+len(join_FP.index)*41.0309))
print("TN: {}+{}={}".format(join_TN["time"].sum()*4,len(join_TN.index),join_TN["time"].sum()*4+len(join_TN.index)*41.0309))
print("FN: {}+{}={}".format(join_FN["time"].sum()*4,len(join_FN.index),join_FN["time"].sum()*4+len(join_FN.index)*41.0309))
print("Tdebug={}".format(join_TN["time"].sum()*4/len(join_FN.index)))
print()
print("({}-Tdebug*{})/Ttotal={}".format(join_TN["time"].sum()*4+len(join_TN.index)*41.0309 - 0.0022465038*len(joint), len(join_FN.index), tm.strftime('%d %H:%M:%S', tm.gmtime(joint["time"].sum()*4+len(joint.index)*41.0309))))
print("TTN={}".format(tm.strftime('%d %H:%M:%S', tm.gmtime(join_TN["time"].sum()*4+len(join_TN.index)*41.0309))))
print("Tpred={}".format(tm.strftime('%d %H:%M:%S', tm.gmtime(0.0022465038*len(joint)))))
print("Save={}".format((join_TN["time"].sum()*4+len(join_TN.index)*41.0309 - 0.0022465038*len(joint))/ (joint["time"].sum()*4+len(joint.index)*41.0309)))
print("FN={}".format(len(join_FN.index)))
print("Total={}".format(len(joint.index)))




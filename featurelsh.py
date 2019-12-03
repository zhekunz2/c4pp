#!/usr/bin/env python

from nearpy.hashes import RandomDiscretizedProjections
import pandas as pd
from numpy import *
from collections import defaultdict
import csv

class featureLsh():
    def __init__(self, stage, bucket):
        self.parentLevel = 5
        self.rdp = RandomDiscretizedProjections('rdp', stage, bucket, rand_seed=98412194)
        self.rdp.reset(5)
        self.hash_dict = {}
        self.data = defaultdict(list)

    def get_hash(self, vector):
        h = self.rdp.hash_vector(vector)[0]
        return h

    def set_hash(self, header):
        self.hash_dict["program"] = "program"
        for i in header:
            key_vec = i.split("_")
            vec = []
            for j in key_vec:
                vec.append(int(j))
            newkey = self.get_hash(vec)
            self.hash_dict[i] = newkey
        print("Setting hash done. Running lsh...")

    def update_dict(self, dicts):
        print("updating_dict")
        for dict in dicts:
            newdict = {}
            for key, value in dict.items():
                newkey = self.hash_dict[key]
                if newkey == "program":
                    newdict[newkey] = value
                else:
                    if not newkey in newdict:
                        if type(value) == str:
                            newdict[newkey] = float(value)
                        else:
                            if isnan(value):
                                newdict[newkey] = 0
                            else:
                                newdict[newkey] = float(value)
                    else:
                        if type(value) == str:
                            newdict[newkey] += float(value)
                        else:
                            if not isnan(value):
                                newdict[newkey] += float(value)
            for key, value in newdict.items():
                self.data[key].append(value)


def read_csv_to_dict(filename):
    csv = pd.read_csv(filename, sep=',', header=None, dtype=str).to_dict()
    df = pd.DataFrame(csv)
    print(df)
    header = []
    for i in range(1, df.shape[1]):
        header.append(df.iloc[0, i])
    mydicts = []
    for i in range(1, df.shape[0]):
        mydict = {}
        mydict["program"] = df.iat[i, 0]
        for j in range(1, df.shape[1]):
            mydict[df.iloc[0, j]] = df.iloc[i, j]
        mydicts.append(mydict)
    return mydicts, header


if __name__ == '__main__':
    b_range = [1]
    for i in range(5, 101, 5):
        b_range.append(i)
    mydicts, header = read_csv_to_dict("timeseries_mutants_7_features_ast_5_raw.csv")
    print("Reading csv done")
    for stage in (1, 5, 10):
        for bucket in b_range:
            mylsh = featureLsh(stage, bucket)
            mylsh.set_hash(header)
            mylsh.update_dict(mydicts)
            print("Outputting")
            tmp = pd.DataFrame.from_dict(mylsh.data)
            tmp.to_csv('output/timeseries_mutants_'+str(stage)+'_features_ast_'+str(bucket)+'_lsh.csv', header=True, index=False)
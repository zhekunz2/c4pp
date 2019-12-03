#!/usr/bin/env python
from nearpy.engine import Engine
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjections,RandomDiscretizedProjections
import sys
import pandas as pd
import numpy as np
import argparse
from scipy.stats import *
from sklearn.covariance import  *


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', dest='feature_file')
    parser.add_argument('-l', nargs='+', dest='metrics')
    parser.add_argument('-k', nargs='+', dest='keep')
    parser.add_argument('-o', dest='output')
    args = parser.parse_args()

    return args


def createLSH(dimensions):
    nearest = NearestFilter(5)
    bin_width = 10
    projections = 50
    rbp = RandomDiscretizedProjections('rbp', projections, bin_width)
    rbp2 = RandomDiscretizedProjections('rbp2', projections, bin_width)
    rbp3 = RandomDiscretizedProjections('rbp3', projections, bin_width)
    rbp4 = RandomDiscretizedProjections('rbp4', projections, bin_width)

    engine = Engine(dimensions, lshashes=[rbp, rbp2, rbp3, rbp4], vector_filters=[nearest])
    return engine


def read_csv(file):
    df = pd.read_csv(file, index_col='program').astype(np.float32)
    df.index = df.index.map(lambda x: x.split('/')[-2] if len(x.split('/')) > 1 else x)
    df = df.replace('inf', np.inf)

    df = df.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
    return df


def read_all_csvs(csv_files, keep=None):
    all_csvs= None
    for file in csv_files:
        df=read_csv(file)

        if file.endswith('features.csv') and keep is not None:
                f_keep = []
                for k in args.keep:
                    f_keep += list(filter(lambda x: x.startswith(k), list(df)))
                df = df.filter(f_keep)
        if all_csvs is None:
            all_csvs = df
        else:
            #all_csvs = all_csvs.merge(df)
            all_csvs = pd.merge(left=all_csvs, right=df, how='outer', left_index=True, right_index=True)
    all_csvs = all_csvs.fillna(0)
    return all_csvs


def mychi2(data, bins=10):
    counts = dict()
    bin_ranges=np.linspace(0, 2, bins)
    for b in bin_ranges:
        counts[b] = 0
    for i in data:
        for b in bin_ranges:
            if i < b:
                counts[b] = counts[b]+1
    #print(counts)
    return chisquare(counts.values())[0]


def get_dataset_balance(df, indices, metric):
    present =None
    if indices is not None:
        present = set(df.index).intersection(set(indices))
        df = df.ix[present]

    #print(np.mean(df['wass_value_avg']))

    print(mychi2(df['wass_value_avg'], bins=5))
    print('True: ', len(df[df[metric] == True]), 'False: ', len(df[df[metric] == False]), 'Missing: ',
          len(indices)-len(present) if present is not None else 0)
    print('Covariance mean: ', np.mean(EmpiricalCovariance().fit(df).covariance_))
    print('Covariance std: ', np.var(EmpiricalCovariance().fit(df).covariance_))


def create_batches(df, metrics, batches=3):
    all_indices = list(df.index)
    b=np.linspace(0, len(all_indices), batches+1).astype(int)
    print(b)
    divs = [df.ix[all_indices[b[i]:b[i+1]]] for i in range(batches)]
    for d in divs:
        get_dataset_balance(metrics, list(d.index), 'wass_result_avg')
    return divs


args=read_args()
df = read_all_csvs(args.feature_file, keep=args.keep)


metrics = read_all_csvs(args.metrics)

print('Original')

get_dataset_balance(metrics, None, 'wass_result_avg')

engine = createLSH(len(list(df)))
batches = create_batches(df, metrics)
selected = []
for k in range(len(batches)):
    if k == 0:
        # store
        for l in list(batches[k].index):
            engine.store_vector(np.array(batches[k].ix[l], dtype=np.float32), data=l.strip())
            selected.append(l)
    else:
        # predict
        for l in list(batches[k].index):
            try:
                N = engine.neighbours(np.array(batches[k].ix[l], dtype=np.float32))
            except:
                print(l)
                print(batches[k].ix[l])
                exit(1)

            if len(N) < 1:
                # can add
                selected.append(l)
                engine.store_vector(np.array(batches[k].ix[l], dtype=np.float32), data=l.strip())

print(len(selected))

get_dataset_balance(metrics, selected, 'wass_result_avg')
if args.output is not None:
    with open(args.output, 'w') as out:
        for s in selected:
            out.write(s+'\n')
exit(1)
#
# for l in test1:
#     engine.store_vector(np.array(df.ix[l.strip()]), data=l.strip())
#
# for l in test2:
#     l = l.strip()
#     N = engine.neighbours(np.array(df.ix[l]))
#     curlabel = metrics.ix[l]['wass_result_avg']
#     pos=0
#     for n in N:
#         neighbour_label=metrics.ix[n[1]]['wass_result_avg']
#         pos+=neighbour_label == True
#
#     if len(N) < 5:
#          print('predicted: False', 'actual: ', curlabel)
#     # else:
#     #      print('predicted: True', 'actual: ',curlabel)

#!/usr/bin/python
import csv
import sys
from scipy.stats import *
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from pandas import *
import numpy as np
import os.path
#from fractions import Fraction
csv.field_size_limit(sys.maxsize)

pydist_dict = {
    "normal" : "norm"
}

rdist_dict = {
    "normal" : "norm"
}

thres_dict = {
    "t" : 0.05,
    "ks" : 0.05,
    "kl" : 1,
    "smkl" : 1,
    "hell" : 0.4,
    "rhat" : 1.001,
    "rhatavg" : 1.01,
    "rhatmax" : 1.01
}

def p_close(p_value, thres):
    if abs(p_value) <= abs(thres):
        return False # p_value stat significant, res different
    else:
        return True

def d_close(divergence, thres):
    if abs(divergence) >= abs(thres):
        return False # divergence too big, res different
    else:
            return True

class DataDataMetric:
    def __init__(self, name, value_a, value_b, **kwargs):
        self.data_a = value_a.sample
        self.rhat_a = value_a.rhat
        self.data_b = value_b.sample
        self.rhat_b = value_b.rhat
        try:
            self.metric = getattr(self, name + "_s")
        except:
            print("Error: unknown metric. Metric must be one of {t, ks, kl, smkl, hell[inger], Rhat}")
            exit(1)

    def reform_data(self, data_a, data_b):
        size = min(len(data_b), len(data_a))
        return (data_a[-size:], data_b[-size:])

    def t_s(self, thres=0):
        # required: data_a and data_b must have the same size
        data_a, data_b = self.reform_data(self.data_a, self.data_b)
        return p_close, ttest_ind(self.data_a, self.data_b)[1]

    def ks_s(self, thres=0):
        return p_close, ks_2samp(self.data_a, self.data_b)[1]

    def kl_s(self, thres=float("inf")):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        r_ret = r('''
                X = as.numeric(t(X))
                Y = as.numeric(t(Y))
                library(FNN)
                kl = na.omit(KL.divergence(X, Y, k = 10, algorithm=c("kd_tree", "cover_tree", "brute")))
                mean(kl[is.infinite(kl) == 0])
                ''')
        r_ret_str = str(r_ret)
        return d_close, float(r_ret_str[4:])

    def smkl_s(self, thres=float("inf")):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        r_ret = r('''
                X = as.numeric(t(X))
                Y = as.numeric(t(Y))
                library(FNN)
                klxy = na.omit(KL.divergence(X, Y, k = 10, algorithm=c("kd_tree", "cover_tree", "brute")))
                klyx = na.omit(KL.divergence(Y, X, k = 10, algorithm=c("kd_tree", "cover_tree", "brute")))
                mean(klxy[is.infinite(klxy) == 0]) + mean(klyx[is.infinite(klyx) == 0])
                ''')
        r_ret_str = str(r_ret)
        return d_close, float(r_ret_str[4:])

    def hell_s(self, thres=1):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        r_ret = r('''
                X = as.numeric(t(X))
                Y = as.numeric(t(Y))
                min2 = min(c(min(X),min(Y)))
                max2 = max(c(max(X),max(Y)))
                library(statip)
                hellinger(X, Y, min2, max2)
                ''')
        r_ret_str = str(r_ret)
        return d_close, float(r_ret_str[4:])

class DataDistMetric:
    def __init__(self, name, value_a, value_b, **kwargs):
        self.data_a = value_a.sample
        self.dist_name = value_b.dist_name
        self.dist_args = value_b.dist_args
        self.close = None
        try:
            self.metric = getattr(self, name + "_s")
        except:
            print("Error: unknown metric. Metric must be one of {t, ks, kl, smkl, hell[inger]}")
            exit(1)

    def pyr_dist(self, pyr):
        try:
            if pyr == "py":
                result = pydist_dict[self.dist_name]
            elif pyr == "r":
                result = rdist_dict[self.dist_name]
        except:
            result = dist_name
        return result

    def dist_obj(self, pydist_name):
        return eval(pydist_name)

    def t_s(self, thres=0):
        return p_close, ttest_1samp(self.data_a, self.dist_obj(self.pyr_dist("py")).mean(*self.dist_args))[1]

    def ks_s(self, thres=0):
        return p_close, kstest(self.data_a, self.pyr_dist("py"), args=self.dist_args)[1]

    def kl_s(self, thres=float("inf")):
        len_data_a = len(self.data_a)
        dict_data_a = dict((x, self.data_a.count(x)/float(len_data_a)) for x in self.data_a)
        keys = dict_data_a.keys()
        p = [dict_data_a[kk] for kk in keys]
        q = self.dist_obj(self.pyr_dist("py")).pdf(keys, *self.dist_args)
        q = [np.finfo(np.float32).eps if qq == 0 else qq for qq in q]
        return d_close, entropy(p, q)

    def smkl_s(self, thres=float("inf")):
        len_data_a = len(self.data_a)
        dict_data_a = dict((x, self.data_a.count(x)/float(len_data_a)) for x in self.data_a)
        keys = dict_data_a.keys()
        p = [dict_data_a[kk] for kk in keys]
        q = self.dist_obj(self.pyr_dist("py")).pdf(keys, *self.dist_args)
        # q = [np.finfo(np.float32).eps if qq == 0 else qq for qq in q]
        return d_close, entropy(p, q) + entropy(q, p)

    def hell_s(self, thres=1):
        r.assign('X', self.data_a)
        r_ret = r('''
                X = as.numeric(t(X))
                Y = r{}({}, {})
                min2 = min(c(min(X),min(Y)))
                max2 = max(c(max(X),max(Y)))
                library(statip)
                hellinger(X, Y, min2, max2)
                '''.format(self.pyr_dist("r"), len(self.data_a),\
                        str(self.dist_args)[1:-1]))
        r_ret_str = str(r_ret)
        return d_close, float(r_ret_str[4:])

    #def wass_s(self, thres=float("inf")):
    #    scipy.stats.wasserstein_distance

class Data:
    def __init__(self, sample, rhat):
        self.sample = sample
        self.rhat = rhat

class Dist:
    def __init__(self, dist_name, dist_args):
        self.dist_name = dist_name
        self.dist_args = dist_args

def fitted(data_str):
    if data_str[0].strip()[0] == '[':
        sample = [float(dd) for dd in data_str[0].strip(' []').split()]
        try:
            rhat = float(data_str[1])
        except:
            rhat = 0
        #return Data(sample[len(sample)/2:], rhat)
        return Data(sample[-1000:], rhat)
    else:
        return Dist(data_str[0].strip().lower(), [float(dd.strip(' []')) for dd in data_str[1:]])

def file_to_dict(file_name):
    with open(file_name) as f:
        data_a_reader = csv.reader(f, delimiter='\n')
        data_a = []
        for data_a_str in data_a_reader:
            data_a.extend(data_a_str)
    dict_a = {}
    for aa in data_a:
        aa_split = aa.split(',')
        dict_a[aa_split[0].strip().lower().replace("[","_").replace("]","")] = fitted(aa_split[1:])
    list(dict_a)[0]
    return dict_a


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        data_file_a = sys.argv[1]
        data_file_b = sys.argv[2]
    else:
        print(\
                "Usage: ./metrics.py csv1 [csv2] [metric threshold]\n\tmetric: one from {t, ks, kl, smkl, hell[inger], rhat[avg]}")
        exit(0)
    try:
        dict_a = file_to_dict(data_file_a)
        dict_b = file_to_dict(data_file_b)
    except:
        result = ["avg"]
        result.extend([str(np.nan)] * 10)
        thres = thres_dict["rhatavg"]
        if 'dict_a' in locals():
            dict_a = file_to_dict(data_file_a)
            all_rhat = []
            for key_a, value in dict_a.iteritems():
                if str(value.rhat) != "nan": all_rhat.append(value.rhat)
            result.extend([str(d_close(np.mean(all_rhat), thres)), str(np.mean(all_rhat))])
            result.extend([str(np.nan)] * 2)
        elif 'dict_b' in locals():
            dict_a = file_to_dict(data_file_b)
            all_rhat = []
            for key_a, value in dict_a.iteritems():
                if str(value.rhat) != "nan": all_rhat.append(value.rhat)
            result.extend([str(np.nan)] * 2)
            result.extend([str(d_close(np.mean(all_rhat), thres)), str(np.mean(all_rhat))])
        else:
            result.extend([str(np.nan)] * 4)
        print(",".join(result))
        exit(0)

    try:
        metrics = [sys.argv[3].replace("avg","")[:4].lower()]
    except:
        metrics = ["t", "ks", "kl", "smkl", "hell"]
    if len(sys.argv) > 4:
        thres_dict[sys.argv[3][:4].lower()] = float(sys.argv[4])
    dict_a = file_to_dict(data_file_a)
    if data_file_b.lower()[:4] == "rhat":
        try:
            thres = float(sys.argv[3])
        except:
            thres = thres_dict[data_file_b.lower()]

        if data_file_b.lower()[-3:] == "avg":
            all_rhat = []
            for key_a, value in dict_a.iteritems():
                if str(value.rhat) != "nan": all_rhat.append(value.rhat)
            print("avg,{},{}".format(d_close(np.mean(all_rhat), thres), np.mean(all_rhat)))
        elif data_file_b.lower()[-3:] == "max":
            all_rhat = []
            for key_a, value in dict_a.iteritems():
                if str(value.rhat) != "nan": all_rhat.append(value.rhat)
            print("max,{},{}".format(d_close(max(all_rhat), thres), max(all_rhat)))
        else:
            for key_a, value in dict_a.iteritems():
                print("{},{},{}".format(key_a.strip(), d_close(value.rhat, thres), value.rhat))
    else:
        dict_b = file_to_dict(data_file_b)
        df = DataFrame()
        param_set = (set([kk.strip() for kk in dict_a.keys()]) & set([kk.strip() for kk in dict_b.keys()]))
        for key_a in param_set:
            result = {"param" : key_a}
            value_a = dict_a[key_a]
            value_b = dict_b[key_a]
            for metric in metrics:
                thres = thres_dict[metric]
                if isinstance(value_a, Data) and isinstance(value_b, Data):
                    dd_metric = DataDataMetric(metric, value_a, value_b)
                elif isinstance(value_a, Data):
                    dd_metric = DataDistMetric(metric, value_a, value_b)
                elif isinstance(value_b, Data):
                    dd_metric = DataDistMetric(metric, value_b, value_a)
                # else:
                #     dd_metric = data_dist_metric(metric.lower(), data_a, data_b)
                is_close, value = dd_metric.metric(thres)
                result[metric + "_value"] = value
                result[metric + "_is_close"] = is_close
            if isinstance(value_a, Data):
                result["rhat1_value"] = dict_a[key_a].rhat
            if isinstance(value_b, Data):
                result["rhat2_value"] = dict_b[key_a].rhat
            df = df.append(Series(result), ignore_index=True)
        if len(metrics) > 1 or sys.argv[3][-3:] == "avg":
            param_set = ["avg"]
        for param in param_set:
            output_str = [param]
            for metric in metrics:
                thres = thres_dict[metric]
                mean = df[metric + "_value"].mean()
                close = df[metric + "_is_close"][0](mean, thres)
                output_str.extend([str(close), str(mean)])
            if "rhat1_value" in list(df) and len(metrics) != 1:
                thres = thres_dict["rhatavg"]
                mean = df["rhat1_value"].mean()
                close = d_close(mean, thres)
                output_str.extend([str(close), str(mean)])
            if "rhat2_value" in list(df) and len(metrics) != 1:
                thres = thres_dict["rhatavg"]
                mean = df["rhat2_value"].mean()
                close = d_close(mean, thres)
                output_str.extend([str(close), str(mean)])
            print(",".join(output_str))
        #if len(metrics) == 1 and sys.argv[3][-3:] == "avg":
        #    print("avg,{},{}".format(is_close(np.mean(all_value), thres), np.mean(all_value)))
        #else:
        #    print(output_str)

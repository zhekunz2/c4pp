#!/usr/bin/env python

# ./metrics_0301.py -c -fs summary_100000 -m rhat -m ess
# ./metrics_0301.py -fc output_1000.csv -fc output_100000_thin.csv -m t -m ks -m kl -m smkl -m hell

import csv
import sys
from scipy.stats import *
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from pandas import *
import pandas as pd
import numpy as np
import os.path
import argparse
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
    "rhat" : 1.01,
    "rhatavg" : 1.01,
    "rhatmax" : 1.01,
    "ess_n" : 0.01
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
    def __init__(self, value_a, value_b):
        self.data_a = value_a.sample
        self.data_b = value_b.sample
        #self.rhat_a = value_a.rhat
        #self.rhat_b = value_b.rhat

    def eval_metrics(self, metrics, thresholds):
        result = []
        for metric_name, threshold in zip(metrics, thresholds):
            try:
                metric = getattr(self, metric_name + "_s")
            except:
                print("Error: unknown metric. Metric must be one of {t, ks, kl, smkl, hell[inger], Rhat}")
                exit(0)
            result.append(metric(threshold))
        return result


    def reform_data(self, data_a, data_b):
        size = min(len(data_b), len(data_a))
        return (data_a[-size:], data_b[-size:])

    def t_s(self, thres=0):
        # required: data_a and data_b must have the same size
        data_a, data_b = self.reform_data(self.data_a, self.data_b)
        statistics = ttest_ind(self.data_a, self.data_b)[1]
        return p_close(statistics, thres), statistics

    def ks_s(self, thres=0):
        statistics = ks_2samp(self.data_a, self.data_b)[1]
        return p_close(statistics, thres), statistics

    def kl_s(self, thres=float("inf")):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        try:
            r_ret = r('''
                    X = as.numeric(t(X))
                    Y = as.numeric(t(Y))
                    library(FNN)
                    kl = na.omit(KL.divergence(X, Y, k = 10, algorithm=c("kd_tree", "cover_tree", "brute")))
                    mean(kl[is.infinite(kl) == 0])
                    ''')
            r_ret_str = str(r_ret)
            statistics = float(r_ret_str[4:])
        except:
            statistics = np.nan
        return d_close(statistics, thres), statistics

    def smkl_s(self, thres=float("inf")):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        try:
            r_ret = r('''
                    X = as.numeric(t(X))
                    Y = as.numeric(t(Y))
                    library(FNN)
                    klxy = na.omit(KL.divergence(X, Y, k = 10, algorithm=c("kd_tree", "cover_tree", "brute")))
                    klyx = na.omit(KL.divergence(Y, X, k = 10, algorithm=c("kd_tree", "cover_tree", "brute")))
                    mean(klxy[is.infinite(klxy) == 0]) + mean(klyx[is.infinite(klyx) == 0])
                    ''')
            r_ret_str = str(r_ret)
            statistics = float(r_ret_str[4:])
        except:
            statistics = np.nan
        return d_close(statistics, thres), statistics

    def hell_s(self, thres=1):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        try:
            r_ret = r('''
                    X = as.numeric(t(X))
                    Y = as.numeric(t(Y))
                    min2 = min(c(min(X),min(Y)))
                    max2 = max(c(max(X),max(Y)))
                    library(statip)
                    hellinger(X, Y, min2, max2)
                    ''')
            r_ret_str = str(r_ret)
            statistics = float(r_ret_str[4:])
        except:
            statistics = np.nan
        return d_close(statistics, thres), statistics

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
    def __init__(self, sample):
        self.sample = sample
        #self.rhat = rhat

class Dist:
    def __init__(self, dist_name, dist_args):
        self.dist_name = dist_name
        self.dist_args = dist_args

class Summary:
    def __init__(self, summary_file):
        self.summary_file = summary_file
        summary_df = pd.read_csv(summary_file,comment='#') # sep="\s+")
        summary_df.set_index("name", inplace=True)
        summary_df.drop([xx for xx in list(summary_df.index) if "__" in xx], inplace=True)
        self.summary_df = summary_df

    def rhat(self, opt="avg", thres=thres_dict["rhat"]):
        if opt == "avg":
            rhat_avg = self.summary_df.R_hat.mean()
            return d_close(rhat_avg, thres),rhat_avg
        elif opt == "max":
            rhat_max = self.summary_df.R_hat.max()
            return d_close(rhat_max, thres),rhat_max
        else:
            print("Error: option for rhat must be \"avg\" or \"max\"")
            exit(0)

    def ess_n(self, opt="avg", thres=thres_dict["ess_n"]):
        import re
        with open(self.summary_file, 'r') as f:
            text=f.read()
            matches = re.findall("(?<=iter=\()\d+", text)
            iters_n = int(matches[0])
        if opt == "avg":
            ess_avg = self.summary_df.N_Eff.mean()
            ess_avg_n = ess_avg/iters_n
            return p_close(ess_avg_n, thres),ess_avg_n
        elif opt == "max":
            ess_max = self.summary_df.N_Eff.max()
            ess_max_n = ess_max/iters_n
            return p_close(ess_max_n, thres),ess_max_n
        else:
            print("Error: option for ess must be \"avg\" or \"max\"")
            exit(0)

class Stan_CSV:
    def __init__(self, csv_file):
        csv_df = pd.read_csv(csv_file,comment='#')
        csv_df.drop([xx for xx in list(csv_df) if "__" in xx], axis=1, inplace=True)
        self.csv_df = csv_df[-1000:]

class Stan_CSV_chains:
    def __init__(self, csv_files):
        csv_dfs = []
        for ff in csv_files:
            csv_df = pd.read_csv(ff,comment='#')
            csv_df.drop([xx for xx in list(csv_df) if "__" in xx], axis=1, inplace=True)
            csv_dfs.append(csv_df[1000:])
        self.csv_df = pd.concat(csv_dfs)

    def get_size(self):
        return len(self.csv_df.index)

    def resize(self, length):
        self.csv_df = self.csv_df[-length:]

# csv_a, csv_b: Stan_CSV dataframes
def csv_metric(csv_a, csv_b, metrics, thresholds):
    csv_df_a = csv_a.csv_df
    csv_df_b = csv_b.csv_df
    param_set = (set([kk.strip() for kk in csv_df_a]) & set([kk.strip() for kk in csv_df_b]))
    result = []
    for param in param_set:
        data_a = Data(list(csv_df_a[param]))
        data_b = Data(list(csv_df_b[param]))
        dd_metric = DataDataMetric(data_a, data_b)
        # for each param a bunch of tuples
        all_result_value = dd_metric.eval_metrics(metrics, thresholds)
        result.append(all_result_value)
    list_list_tuples = map(list, zip(*result))
    all_result_stats = []
    for test in list_list_tuples:
        test_result = map(list, zip(*test))[0]
        test_stats = map(list, zip(*test))[1]
        all_result_stats.append((all(test_result), np.mean(test_stats)))
    return all_result_stats

    #    param_set = (set([kk.strip() for kk in dict_a.keys()]) & set([kk.strip() for kk in dict_b.keys()]))
    #    for key_a in param_set:
    #        results = {"param" : key_a}
    #        value_a = dict_a[key_a]
    #        value_b = dict_b[key_a]
    #        for metric in metrics:
    #            thres = thres_dict[metric]
    #            if isinstance(value_a, Data) and isinstance(value_b, Data):
    #                dd_metric = DataDataMetric(metric, value_a, value_b)
    #            elif isinstance(value_a, Data):
    #                dd_metric = DataDistMetric(metric, value_a, value_b)
    #            elif isinstance(value_b, Data):
    #                dd_metric = DataDistMetric(metric, value_b, value_a)
    #            # else:
    #            #     dd_metric = data_dist_metric(metric.lower(), data_a, data_b)
    #            is_close, value = dd_metric.metric(thres)
    #            results[metric + "_value"] = value
    #            results[metric + "_is_close"] = is_close
    #        if isinstance(value_a, Data):
    #            results["rhat1_value"] = dict_a[key_a].rhat
    #        if isinstance(value_b, Data):
    #            results["rhat2_value"] = dict_b[key_a].rhat
    #        df = df.append(Series(results), ignore_index=True)



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
    pp = argparse.ArgumentParser()
    pp.add_argument("-fc", "--csv_file", action="append", default=[],
            help="CSV data file(s) to use")
    pp.add_argument("-fs", "--summary_file", action="append", default=[],
            help="Stan summary file(s) to use")
    pp.add_argument("-fm", "--min_files", action="append", default=[],
            help="CSV data file(s) for minimum iters in .gz")
    pp.add_argument("-fr", "--ref_files", action="append", default=[],
            help="CSV data file(s) for 100000 iters in .gz")
    pp.add_argument("-fp", "--param_file", action="append", default=[],
            help="Formatted param file(s) with samples and rhat")
    pp.add_argument("-c", action="store_true", default=False,
            dest="conv", help="Calculate convergence metrics instead of \
                    accuracy metrics")
    pp.add_argument("-m", "--metric", action="append", default=[],
            help="Metric to calculate.\n If CSV file is provided, the metric\
                    must be one from {t, ks, kl, smkl, hell[inger]; \n\
                    If Stan summary file is provided, the metric must be one\n\
                    from {rhat, avg}.")
    pp.add_argument("-t", "--threshold", action="append", default=[],
            help="Set customer threshold")
    args = pp.parse_args()
    # print("csv_file")
    # print(args.csv_file)
    # print("metric")
    # print(args.metric)
    # print("conv")
    # print(args.conv)
    # print("thres")
    # print(args.threshold)
    if args.conv:
        if args.summary_file:
            # ./metrics_0301.py -c -fs summary_100000 -m rhat
            data_file_a = args.summary_file[0]
            summary = Summary(data_file_a)
            rhat_ess_ret = []
            if "rhat" in args.metric:
                if len(args.threshold) == 0:
                    rhat_ess_ret.append(str(summary.rhat())[1:-1])
                else:
                    rhat_ess_ret.append(str(summary.rhat(thres=float(args.threshold[0])))[1:-1])
            if "ess" in args.metric:
                rhat_ess_ret.append(str(summary.ess_n())[1:-1])
            if len(rhat_ess_ret) != 0:
                print(", ".join(rhat_ess_ret))

    else:
        if args.csv_file:
            # ./metrics_0301.py -fc output_1000.csv -fc output_100000_thin.csv -m t
            data_file_a = args.csv_file[0]
            data_file_b = args.csv_file[1]
            if len(args.threshold) > 0:
                try:
                    thresholds = map(int, args.threshold)
                except:
                    print("Error: invalid threshold")
                    exit(0)
            else:
                thresholds = [thres_dict[mm] for mm in args.metric]
            stan_csv_a = Stan_CSV(data_file_a)
            stan_csv_b = Stan_CSV(data_file_b)
            metric_ret = csv_metric(stan_csv_a, stan_csv_b, args.metric, thresholds)
            print(", ".join([str(mm)[1:-1] for mm in metric_ret]))
        elif args.ref_files and args.min_files:
            thresholds = [thres_dict[mm] for mm in args.metric]
            stan_csv_ref = Stan_CSV_chains(args.ref_files)
            stan_csv_min = Stan_CSV_chains(args.min_files)
            stan_csv_ref.resize(stan_csv_min.get_size())
            metric_ret = csv_metric(stan_csv_ref, stan_csv_min, args.metric, thresholds)
            print(", ".join([str(mm)[1:-1] for mm in metric_ret]))



    #if len(sys.argv) >= 3:
    #    data_file_a = sys.argv[1]
    #    data_file_b = sys.argv[2]
    #else:
    #    print(\
    #            "Usage: ./metrics.py csv1 [csv2] [metric threshold]\n\tmetric: one from {t, ks, kl, smkl, hell[inger], rhat[avg]}")
    #    exit(0)
    #try:
    #    dict_a = file_to_dict(data_file_a)
    #    dict_b = file_to_dict(data_file_b)
    #except:
    #    results = ["avg"]
    #    results.extend([str(np.nan)] * 10)
    #    thres = thres_dict["rhatavg"]
    #    if 'dict_a' in locals():
    #        dict_a = file_to_dict(data_file_a)
    #        all_rhat = []
    #        for key_a, value in dict_a.iteritems():
    #            if str(value.rhat) != "nan": all_rhat.append(value.rhat)
    #        results.extend([str(d_close(np.mean(all_rhat), thres)), str(np.mean(all_rhat))])
    #        results.extend([str(np.nan)] * 2)
    #    elif 'dict_b' in locals():
    #        dict_a = file_to_dict(data_file_b)
    #        all_rhat = []
    #        for key_a, value in dict_a.iteritems():
    #            if str(value.rhat) != "nan": all_rhat.append(value.rhat)
    #        results.extend([str(np.nan)] * 2)
    #        results.extend([str(d_close(np.mean(all_rhat), thres)), str(np.mean(all_rhat))])
    #    else:
    #        results.extend([str(np.nan)] * 4)
    #    print(",".join(results))
    #    exit(0)

    #try:
    #    metrics = [sys.argv[3].replace("avg","")[:4].lower()]
    #except:
    #    metrics = ["t", "ks", "kl", "smkl", "hell"]
    #if len(sys.argv) > 4:
    #    thres_dict[sys.argv[3][:4].lower()] = float(sys.argv[4])
    #dict_a = file_to_dict(data_file_a)
    #if data_file_b.lower()[:4] == "rhat":
    #    try:
    #        thres = float(sys.argv[3])
    #    except:
    #        thres = thres_dict[data_file_b.lower()]

    #    if data_file_b.lower()[-3:] == "avg":
    #        all_rhat = []
    #        for key_a, value in dict_a.iteritems():
    #            if str(value.rhat) != "nan": all_rhat.append(value.rhat)
    #        print("avg,{},{}".format(d_close(np.mean(all_rhat), thres), np.mean(all_rhat)))
    #    elif data_file_b.lower()[-3:] == "max":
    #        all_rhat = []
    #        for key_a, value in dict_a.iteritems():
    #            if str(value.rhat) != "nan": all_rhat.append(value.rhat)
    #        print("max,{},{}".format(d_close(max(all_rhat), thres), max(all_rhat)))
    #    else:
    #        for key_a, value in dict_a.iteritems():
    #            print("{},{},{}".format(key_a.strip(), d_close(value.rhat, thres), value.rhat))
    #else:
    #    dict_b = file_to_dict(data_file_b)
    #    df = DataFrame()
    #    param_set = (set([kk.strip() for kk in dict_a.keys()]) & set([kk.strip() for kk in dict_b.keys()]))
    #    for key_a in param_set:
    #        results = {"param" : key_a}
    #        value_a = dict_a[key_a]
    #        value_b = dict_b[key_a]
    #        for metric in metrics:
    #            thres = thres_dict[metric]
    #            if isinstance(value_a, Data) and isinstance(value_b, Data):
    #                dd_metric = DataDataMetric(metric, value_a, value_b)
    #            elif isinstance(value_a, Data):
    #                dd_metric = DataDistMetric(metric, value_a, value_b)
    #            elif isinstance(value_b, Data):
    #                dd_metric = DataDistMetric(metric, value_b, value_a)
    #            # else:
    #            #     dd_metric = data_dist_metric(metric.lower(), data_a, data_b)
    #            is_close, value = dd_metric.metric(thres)
    #            results[metric + "_value"] = value
    #            results[metric + "_is_close"] = is_close
    #        if isinstance(value_a, Data):
    #            results["rhat1_value"] = dict_a[key_a].rhat
    #        if isinstance(value_b, Data):
    #            results["rhat2_value"] = dict_b[key_a].rhat
    #        df = df.append(Series(results), ignore_index=True)
    #    if len(metrics) > 1 or sys.argv[3][-3:] == "avg":
    #        param_set = ["avg"]
    #    for param in param_set:
    #        output_str = [param]
    #        for metric in metrics:
    #            thres = thres_dict[metric]
    #            mean = df[metric + "_value"].mean()
    #            close = df[metric + "_is_close"][0](mean, thres)
    #            output_str.extend([str(close), str(mean)])
    #        if "rhat1_value" in list(df) and len(metrics) != 1:
    #            thres = thres_dict["rhatavg"]
    #            mean = df["rhat1_value"].mean()
    #            close = d_close(mean, thres)
    #            output_str.extend([str(close), str(mean)])
    #        if "rhat2_value" in list(df) and len(metrics) != 1:
    #            thres = thres_dict["rhatavg"]
    #            mean = df["rhat2_value"].mean()
    #            close = d_close(mean, thres)
    #            output_str.extend([str(close), str(mean)])
    #        print(",".join(output_str))
    #    #if len(metrics) == 1 and sys.argv[3][-3:] == "avg":
    #    #    print("avg,{},{}".format(is_close(np.mean(all_value), thres), np.mean(all_value)))
    #    #else:
    #    #    print(output_str)

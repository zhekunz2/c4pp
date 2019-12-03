#!/usr/bin/env python

# ~/c4pp/metrics/metrics_0301.py -c -fs rw_summary_1000 -m rhat -m ess -o avg -o extreme -rt
# ~/c4pp/metrics/metrics_0301.py -fr output_100000.gz -fm output_1000_1.gz -fm output_1000_2.gz -fm output_1000_3.gz -fm output_1000_4.gz  -m kl  -m ks -s 200 -s 100 -l 100
# ./metrics_0301.py -fc output_1000.csv -fc output_100000_thin.csv -m t -m ks -m kl -m smkl -m hell
# ./metrics_0301.py -fm output_1000.csv -fr output_100000_thin.csv -m t -m ks -m kl  -t 0.6 -t 0.5 -t 9

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
import re
#from fractions import Fraction
csv.field_size_limit(sys.maxsize)
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

close_dict = {
    "t": p_close,
    "ks": p_close,
    "kl": d_close,
    "smkl": d_close,
    "hell": d_close
}

extreme_dict = {
    "t": min,
    "ks": min,
    "kl": max,
    "smkl": max,
    "hell": max,
}

class DataDataMetric:
    def __init__(self, value_a, value_b):
        self.data_a = value_a.sample
        self.data_b = value_b.sample
        #self.rhat_a = value_a.rhat
        #self.rhat_b = value_b.rhat

    def eval_metrics(self, metrics, thresholds, var_check = False):
        result = []
        for metric_name, threshold in zip(metrics, thresholds):
            try:
                metric = getattr(self, metric_name + "_s")
            except:
                print("Error: unknown metric. Metric must be one of {t, ks, kl, smkl, hell[inger], Rhat}")
                exit(0)
            if "kl" in metric_name and var_check:
                result.append(metric(threshold) + (metric_name,))
                result.append(metric(threshold, var_check = True) + (metric_name,))
            else:
                result.append(metric(threshold) + (metric_name,))
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

    def var_small(self):
        var_a = np.var(self.data_a)
        var_b = np.var(self.data_b)
        diff_ab = abs(np.mean(self.data_a) - np.mean(self.data_b))
        return diff_ab < 0.01 and var_a < 0.01 and var_b < 0.01

    def kl_s(self, thres=float("inf"), var_check=False):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        if (self.data_a == self.data_b):
            statistics = 0
        elif var_check and self.var_small():
            statistics = 0
        else:
            try:
                r_ret = r('''
                        X = as.numeric(t(X))
                        Y = as.numeric(t(Y))
                        X = na.omit(X)
                        Y = na.omit(Y)
                        library(FNN)
                        kl = KL.divergence(X, Y, k = 20, algorithm=c("kd_tree", "cover_tree", "brute"))
                        if(length(kl[is.finite(kl)]) == 0) {Inf} else { mean(kl[is.finite(kl)]) }
                        ''')
                r_ret_str = str(r_ret)
                statistics = float(r_ret_str[4:])
                if statistics < 0:
                    statistics = 0
            except:
                statistics = np.nan
        return d_close(statistics, thres), statistics

    def smkl_s(self, thres=float("inf"), var_check=False):
        r.assign('X', self.data_a)
        r.assign('Y', self.data_b)
        if (self.data_a == self.data_b):
            statistics = 0
        elif var_check and self.var_small():
            statistics = 0
        else:
            try:
                r_ret = r('''
                        X = as.numeric(t(X))
                        Y = as.numeric(t(Y))
                        X = na.omit(X)
                        Y = na.omit(Y)
                        library(FNN)
                        klxy = KL.divergence(X, Y, k = 20, algorithm=c("kd_tree", "cover_tree", "brute"))
                        klyx = KL.divergence(Y, X, k = 20, algorithm=c("kd_tree", "cover_tree", "brute"))
                        if(length(klxy[is.finite(klxy)]) == 0 | length(klyx[is.finite(klyx)]) == 0) {Inf} else{
                            mean(klxy[is.finite(klxy)]) + mean(klyx[is.finite(klyx)])
                        }
                        ''')
                r_ret_str = str(r_ret)
                statistics = float(r_ret_str[4:])
                if statistics < 0:
                    statistics = 0
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
            statistics = np.inf
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
    def __init__(self, summary_file, runtime=False):
        self.summary_file = summary_file
        self.runtime_df = None
        summary_df = pd.read_csv(summary_file,comment='#') # sep="\s+")
        summary_df.set_index("name", inplace=True)
        if not runtime:
            summary_df.drop([xx for xx in list(summary_df.index) if "__" in xx], inplace=True)
            self.summary_df = summary_df
        else:
            self.summary_df = summary_df.drop([xx for xx in list(summary_df.index) if "__" in xx])
            self.runtime_df = summary_df.drop([xx for xx in list(summary_df.index) if "__" not in xx])

    def rhat(self, thres=thres_dict["rhat"], opt=["avg"]):
        ret = []
        for oo in opt:
            if "av" in oo:
                rhat_avg = self.summary_df.R_hat.mean()
                ret += [d_close(rhat_avg, thres),rhat_avg]
            elif "ext" in oo:
                rhat_ext = self.summary_df.R_hat.max()
                ret += [d_close(rhat_ext, thres),rhat_ext]
        return ret
        #else:
        #    print("Error: option for rhat must be \"avg\" or \"max\"")
        #    exit(0)

    def ess_n(self, thres=thres_dict["ess_n"], opt=["avg"]):
        import re
        ret = []
        with open(self.summary_file, 'r') as f:
            text=f.read()
            matches = re.findall("(?<=iter=\()\d+", text)
            iters_n = int(matches[0])
        ret = []
        for oo in opt:
            if "av" in oo:
                ess_avg = self.summary_df.N_Eff.mean()
                ess_avg_n = ess_avg/iters_n
                ret += [p_close(ess_avg_n, thres),ess_avg_n]
            elif "ext" in oo:
                ess_ext = self.summary_df.N_Eff.min()
                ess_ext_n = ess_ext/iters_n
                ret += [p_close(ess_ext_n, thres),ess_ext_n]
        return ret
        #else:
        #    print("Error: option for ess must be \"avg\" or \"max\"")
        #    exit(0)

    def diagnostics(self, opt = "avg"):
        return ",".join([str(vv) for vv in self.runtime_df.Mean.values])

class Stan_CSV:
    def __init__(self, csv_file):
        csv_df = pd.read_csv(csv_file,comment='#')
        csv_df.drop([xx for xx in list(csv_df) if "__" in xx], axis=1, inplace=True)
        csv_df = csv_df.loc[:, csv_df.mean().apply(np.isfinite)]
        self.csv_df = csv_df[-1000:]

class Stan_CSV_chains:
    def __init__(self, csv_files):
        self.csv_dfs = []
        for ff in csv_files:
            csv_df = pd.read_csv(ff,comment='#')
            csv_df.drop([xx for xx in list(csv_df) if "__" in xx], axis=1, inplace=True)
            csv_df = csv_df.loc[:, csv_df.mean().apply(np.isfinite)]
            self.csv_dfs.append(csv_df)

    def concat_dfs(self, warmup=1000, iters=-1, last=-1):
        csv_dfs = []
        for cc in self.csv_dfs:
            if iters == -1:
                csv_dfs.append(cc[warmup:])
            else:
                if last == -1:
                    csv_dfs.append(cc[warmup:warmup+iters])
                else:
                    full_cc = cc[warmup:warmup+iters]
                    csv_dfs.append(full_cc[-last:])

        self.csv_df = pd.concat(csv_dfs)

    def get_size(self):
        return len(self.csv_df.index)

    def resize(self, length):
        # resize by taking the LAST #length samples
        self.csv_df = self.csv_df[-length:]


def csv_metric_pyro(ref_stan, pyro_res, metrics, thresholds, opt=[], var_check=False):
    csv_df_a = ref_stan.csv_df
    param_set = (set([kk.strip() for kk in csv_df_a]))
    result=[]
    if var_check:
        var_check_ret = False
    for param in param_set:
        data_a = Data(list(csv_df_a[param]))
        param_pyro = param.split('.')[0]
        #print(param)
        if param_pyro not in pyro_res:
            #print("skipping .. " + param_pyro)
            continue
        indices=re.findall('\.', param)

        #print(param_pyro)
        if len(indices) == 0:
            data_b = Data(list(getSamples(pyro_res, param_pyro, None)))
        elif len(indices) == 1:
            data_b = Data(list(getSamples(pyro_res, param_pyro, int(param.split('.')[1])-1 )))
        else:
            # assuming 2d
            data_b = Data(list(getSamples(pyro_res, param_pyro, (int(param.split('.')[1]) - 1, int(param.split('.')[2])-1))))
        if 'nan' in data_b.sample:
            continue
        #print(data_b.sample)
        dd_metric = DataDataMetric(data_a, data_b)
        # for each param a bunch of tuples
        all_result_value = dd_metric.eval_metrics(metrics, thresholds, var_check)
        result.append(all_result_value)
        if var_check and not var_check_ret:
            var_check_ret = var_check_ret or dd_metric.var_small()
    list_list_tuples = map(list, zip(*result))
    all_result_stats = []
    for idx, test in enumerate(list_list_tuples):
        test_result = map(list, zip(*test))[0]
        test_stats = map(list, zip(*test))[1]
        test_name = map(list,zip(*test))[2]
        close = close_dict[test_name[0]]
        for oo in opt:
            if "ext" in oo:
                extreme = extreme_dict[test_name[0]]
                #all_result_stats.append((all(test_result), np.mean(test_stats)))
                all_result_stats.append((close(extreme(test_stats),thresholds[idx]), extreme(test_stats)))
            elif "av" in oo:
                all_result_stats.append((close(np.mean(test_stats),thresholds[idx]), np.mean(test_stats)))
    if var_check:
        all_result_stats.append(str([var_check_ret]))
    return all_result_stats


# csv_a, csv_b: Stan_CSV dataframes
def csv_metric(csv_a, csv_b, metrics, thresholds, opt=[], var_check=False):
    csv_df_a = csv_a.csv_df
    csv_df_b = csv_b.csv_df
    param_set = (set([kk.strip() for kk in csv_df_a]) & set([kk.strip() for kk in csv_df_b]))
    result = []
    if var_check:
        var_check_ret = False
    for param in param_set:
        data_a = Data(list(csv_df_a[param]))
        data_b = Data(list(csv_df_b[param]))
        dd_metric = DataDataMetric(data_a, data_b)
        # for each param a bunch of tuples
        all_result_value = dd_metric.eval_metrics(metrics, thresholds, var_check)
        result.append(all_result_value)
        if var_check and not var_check_ret:
            var_check_ret = var_check_ret or dd_metric.var_small()
    list_list_tuples = map(list, zip(*result))
    all_result_stats = []
    if var_check:
        if "kl" in metrics:
            kl_index = metrics.index("kl")
            thresholds.insert(kl_index, thresholds[kl_index])
        if "smkl" in metrics:
            kl_index = metrics.index("smkl")
            thresholds.insert(kl_index, thresholds[kl_index])
    for idx, test in enumerate(list_list_tuples):
        test_result = map(list, zip(*test))[0]
        test_stats = map(list, zip(*test))[1]
        test_name = map(list,zip(*test))[2]
        close = close_dict[test_name[0]]
        for oo in opt:
            if "ext" in oo:
                extreme = extreme_dict[test_name[0]]
                #all_result_stats.append((all(test_result), np.mean(test_stats)))
                all_result_stats.append((close(extreme(test_stats),thresholds[idx]), extreme(test_stats)))
            elif "av" in oo:
                all_result_stats.append((close(np.mean(test_stats),thresholds[idx]), np.mean(test_stats)))
    if var_check:
        all_result_stats.append(str([var_check_ret]))
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


def parse_pyro_samples(samplesfile):
    import ast
    data = {}
    file = open(samplesfile).read().splitlines()
    for f in file:
        name = f.split(':')[0]
        samples = f.split(':')[1]
        cur_arr = np.array(ast.literal_eval(samples.replace('nan', '\"nan\"').replace('inf', "\"inf\"")))
        data[name] = cur_arr
    return data


def getSamples(data, name, indices):
    # e.g getSamples('sigma', None) -- scalar
    # e.g getSamples('sigma', 1) -- 1d
    # e.g getSamples('sigma', (2,1)) -- 2d array
    try:
        if indices is None:
            samples = [x[0] for x in data[name]]
        elif type(indices) == np.int or len(indices) == 1:
            d = data[name]
            samples = [x[indices] for x in d]
        elif len(indices) == 2:
            d = data[name]
            samples = [x[indices[0]][indices[1]] for x in d]
        else:
            samples = []
    except Exception as e:
        samples = ['nan']*1000

    return samples

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
    pp.add_argument("-fpyro", "--pyro_file", action="append", default=[],
                    help="Samples file in pyro")
    pp.add_argument("-c", action="store_true", default=False,
            dest="conv", help="Calculate convergence metrics instead of \
                    accuracy metrics")
    pp.add_argument("-m", "--metric", action="append", default=[],
            help="Metric to calculate.\n If CSV file is provided, the metric\
                    must be one from {t, ks, kl, smkl, hell[inger]; \n\
                    If Stan summary file is provided, the metric must be one\n\
                    from {rhat, ess}.")
    pp.add_argument("-t", "--threshold", action="append", default=[],
            help="Set customer threshold")
    pp.add_argument("-o", "--option", action="append", default=[],
            help="Take the average value or extreme value among all the params\n\
                    must be one from {avg,ext}\n\
                    output would be in the order m1_o1,m1_o2,m2_o1,m2_o2")
    pp.add_argument("-s", "--sample_size", action="append", default=[],
            help="Only use the first #iters from the minimum .gz file\n\
                    with warmup removed.\n\
                    If multiple sample size is calculated, must provide them\n \
                    in a descending order!")
    pp.add_argument("-w", "--warmup", type=int, default=-1,
            help="Delete the warmup samples from all .gz files")
    pp.add_argument("-l", "--last", type=int, default=-1,
            help="Only take the last number of samples for comparison")
    pp.add_argument("-rt", "--runtime", action="store_true", default=False,
            help="Calculate runtime features from Stan summary file")
    pp.add_argument("-vc", "--var_check", action="store_true", default=False,
            help="Add check for small variance but similar mean value")
    args = pp.parse_args()
    if len(args.option) == 0:
        args.option = ["avg"]
    if args.conv:
        if args.summary_file:
            # ./metrics_0301.py -c -fs summary_100000 -m rhat
            data_file_a = args.summary_file[0]
            if not args.runtime:
                metrics = args.metric
                summary = Summary(data_file_a)
                rhat_ess_ret = []
                if "rhat" in metrics:
                    if len(args.threshold) == 0:
                        rhat_ess_ret.append(str(summary.rhat(opt=args.option))[1:-1])
                    else:
                        rhat_ess_ret.append(str(summary.rhat(thres=float(args.threshold[0]), opt=args.option))[1:-1])
                if "ess" in metrics:
                    rhat_ess_ret.append(str(summary.ess_n(opt=args.option))[1:-1])
                if len(rhat_ess_ret) != 0:
                    print(", ".join(rhat_ess_ret))
            else:
                summary = Summary(data_file_a, runtime=True)
                rhat_ess_ret = []
                if len(args.threshold) == 0:
                    rhat_ess_ret.append(str(summary.rhat(opt=args.option))[1:-1])
                else:
                    rhat_ess_ret.append(str(summary.rhat(thres=float(args.threshold[0]),opt=args.option))[1:-1])
                rhat_ess_ret.append(str(summary.ess_n(opt=args.option))[1:-1])
                metrics = list(summary.runtime_df)
                rhat_ess_ret.append(summary.diagnostics())
                if len(rhat_ess_ret) != 0:
                    print(", ".join(rhat_ess_ret))

            #data_file_a = args.summary_file[0]
            #else:
            #    args.me

    else:
        if args.csv_file:
            # ./metrics_0301.py -fc output_1000.csv -fc output_100000_thin.csv -m t
            data_file_a = args.csv_file[0]
            data_file_b = args.csv_file[1]
            if len(args.threshold) > 0:
                if len(args.threshold) != len(args.metric):
                    print("Error: threshold not specified for every metric")
                    exit(0)
                try:
                    thresholds = map(float, args.threshold)
                except:
                    print("Error: invalid threshold")
                    exit(0)
            else:
                thresholds = [thres_dict[mm] for mm in args.metric]
            stan_csv_a = Stan_CSV(data_file_a)
            stan_csv_b = Stan_CSV(data_file_b)
            metric_ret = csv_metric(stan_csv_a, stan_csv_b, args.metric, thresholds, args.option, args.var_check)
            print(", ".join([str(mm)[1:-1] for mm in metric_ret]))
        elif args.ref_files and args.min_files:
            if len(args.threshold) > 0:
                if len(args.threshold) != len(args.metric):
                    print("Error: threshold not specified for every metric")
                    exit(0)
                try:
                    thresholds = map(float, args.threshold)
                except:
                    print("Error: invalid threshold")
                    exit(0)
            else:
                thresholds = [thres_dict[mm] for mm in args.metric]
            stan_csv_ref = Stan_CSV_chains(args.ref_files)
            stan_csv_ref.concat_dfs()
            stan_csv_min = Stan_CSV_chains(args.min_files)
            if args.sample_size:
                for ss in args.sample_size:
                    ss_int = int(ss)
                    stan_csv_min.concat_dfs(iters=ss_int, last=args.last)
                    stan_csv_ref.resize(stan_csv_min.get_size())
                    metric_ret = csv_metric(stan_csv_ref, stan_csv_min, args.metric, thresholds, args.option, args.var_check)
                    print(", ".join([str(mm)[1:-1] for mm in metric_ret]))
            else:
                stan_csv_min.concat_dfs()
                stan_csv_ref.resize(stan_csv_min.get_size())
                metric_ret = csv_metric(stan_csv_ref, stan_csv_min, args.metric, thresholds, args.option, args.var_check)
                print(", ".join([str(mm)[1:-1] for mm in metric_ret]))
        elif args.ref_files and args.pyro_file:
            thresholds = [thres_dict[mm] for mm in args.metric]
            stan_csv_ref = Stan_CSV_chains(args.ref_files)
            stan_csv_ref.concat_dfs()
            stan_csv_ref.resize(1000)
            pyro_results = parse_pyro_samples(args.pyro_file[0])
            metric_ret = csv_metric_pyro(stan_csv_ref, pyro_results, args.metric, thresholds)
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

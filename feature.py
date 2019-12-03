#!/usr/bin/env python
# ./$feature_file [stan_file] [result_file] [data_file] [output_file] [templatefile]? [pyrofile]?

import json
import re

import antlr4
import pandas as pd
import rpy2.robjects as ro
import numpy as np
import sys
import os
from antlr4 import *
from numpy import *
from pandas.errors import EmptyDataError
from scipy.stats import *
from parser.StanLexer import StanLexer
from parser.StanListener import StanListener
from parser.StanParser import StanParser
from ModelBuilder import ModelBuilder
# from feature_pyro import PyroFeatureParser
import traceback

pydist_dict = {
    "normal" : "norm",
    "inv_gamma" : "invgamma",
    "student_t" : "t",
    "double_exponential" : "laplace",
    "gumbel" : "gumbel_r",
    "lognormal" : "lognorm",
    "chi_square" : "chisquare",
    #"inv_chi_square"
    #"bernoulli_logit" not used in prior
    "binomial" : "binom",
    #"beta_binomial"
    "neg_binomial" : "nbinom",
    "hypergeometric" : "hypergeom",
}

disc_dict = {
    "neg_binomial" : "disc",
    "binomial" : "disc",
    "bernoulli" : "disc",
    "categorical" : "disc",
    "hypergeometric" : "disc",
}


class StanFeatureParser(StanListener):
    def __init__(self):
        self.feature_vector = {}
        self.cur_block = ''
        self.data_items = []
        self.curr_loop_deg = 0
        self.cur_loop_count = None
        self.stack = []
        self.parameters = []
        self.stack.append(StanParser.ProgramContext)
        self.sink = None
        self.arithComplexity = 0
        self.observe_variables = []

    def updateVector(self, key):
        if key in self.feature_vector:
            self.feature_vector[key] += 1
        else:
            self.feature_vector[key] = 1
    
    def updateVectorMax(self, key, cur):
        if not key in self.feature_vector:
            self.feature_vector[key] = cur
        elif self.feature_vector[key] < cur:
            self.feature_vector[key] = cur

    def enterDatablk(self, ctx):
        self.cur_block = 'data'

    def exitDatablk(self, ctx):
        self.cur_block = ''

    def enterDecl(self, ctx):
        id = ctx.ID().getText()
        if self.cur_block == 'data':
            data_type = ctx.dtype().getText()
            self.updateVector('d_' + data_type)
            self.curr_data_dim = 0
            self.data_items.append(id)
        elif self.cur_block == 'parameters':
            self.updateVector('param')
            param_type = ctx.dtype().getText()
            self.updateVector(param_type)
            self.curr_param_dim = 0
            self.parameters.append(id)
        elif self.cur_block == 'model':
            self.updateVector('internal')

    def exitDecl(self, ctx):
        if self.cur_block == 'data':
            if 'max_data_dim' in self.feature_vector:
                self.feature_vector['max_data_dim'] = max(self.feature_vector['max_data_dim'], self.curr_data_dim)
            else:
                self.feature_vector['max_data_dim'] = self.curr_data_dim
            self.curr_data_dim = 0
        elif self.cur_block == 'parameters':
            if 'max_param_dim' in self.feature_vector:
                self.feature_vector['max_param_dim'] = max(self.feature_vector['max_param_dim'], self.curr_param_dim)
            else:
                self.feature_vector['max_param_dim'] = self.curr_param_dim
            self.curr_param_dim = 0

    def enterDims(self, ctx):
        if self.cur_block == 'data':
            self.curr_data_dim += len(ctx.dim())
        if self.cur_block == 'parameters':
            self.curr_param_dim += len(ctx.dim())

    def enterFunction_call(self, ctx):
        self.stack.append(ctx)
        self.updateVector('f_'+ctx.inbuilt().getText())

    def exitFunction_call(self, ctx):
        self.stack.pop()

    def enterDistribution_exp(self, ctx):
        dist = ctx.ID().getText()

        # categories
        if dist == 'dirichlet':
            if self.cur_loop_count is not None:
                self.feature_vector["categories"] = self.cur_loop_count

        self.stack.append(ctx)
        self.updateVector(dist)
        if dist in disc_dict:
            self.updateVector("disc")
        if self.sink is not None and self.sink in self.data_items:
            self.updateVector('d_'+dist)
            self.updateVector('observe')
            self.observe_variables.append(self.sink)

    def exitDistribution_exp(self, ctx):
        self.stack.pop()
        dist = ctx.ID().getText()
        if self.sink is not None and self.sink not in self.data_items:
            try:
                dist_param = [eval(ee.getText()) for ee in ctx.expression()]
                if dist in pydist_dict:
                    pydist = pydist_dict[dist]
                else:
                    pydist = dist
                if 'var_max' in self.feature_vector:
                    self.feature_vector['var_max'] = max(self.feature_vector['var_max'], eval(pydist).var(*dist_param))
                else:
                    self.feature_vector['var_max'] = eval(pydist).var(*dist_param)
                if 'var_min' in self.feature_vector:
                    self.feature_vector['var_min'] = min(self.feature_vector['var_min'], eval(pydist).var(*dist_param))
                else:
                    self.feature_vector['var_min'] = eval(pydist).var(*dist_param)
            except:
                pass
                # if 'var_max' not in self.feature_vector:
                #     self.feature_vector['var_max'] = 0
                # if 'var_min' not in self.feature_vector:
                #     self.feature_vector['var_min'] = 2**100

    def enterId_access(self, ctx):
        id = ctx.ID().getText()
        if isinstance(self.stack[-1], StanParser.Distribution_expContext) and id in self.parameters:
            if self.sink is not None and self.sink in self.parameters:
                self.updateVector('dependent_prior')

    def enterArray_access(self, ctx):
        id = ctx.ID().getText()
        if isinstance(self.stack[-1], StanParser.Distribution_expContext) and id in self.parameters:
            if self.sink is not None and self.sink in self.parameters:
                self.updateVector('dependent_prior')

    def enterModelblk(self, ctx):
        self.cur_block = 'model'

    def exitModelblk(self, ctx):
        self.cur_block = ''

    def enterParamblk(self, ctx):
        self.cur_block = 'parameters'

    def exitParamblk(self, ctx):
        self.cur_block = ''

    def enterFor_loop_stmt(self, ctx):
        self.updateVector('loop')
        self.curr_loop_deg += 1
        loopIndex = ctx.range_exp().expression(1).getText()
        if loopIndex in self.data_items:
            self.cur_loop_count = loopIndex
        if 'loop_deg' in self.feature_vector:
            self.feature_vector['loop_deg'] = max(self.feature_vector['loop_deg'], self.curr_loop_deg)
        else:
            self.feature_vector['loop_deg'] = 1

    def exitFor_loop_stmt(self, ctx):
        self.curr_loop_deg -= 1
        self.cur_loop_count = None

    def enterIf_stmt(self, ctx):
        self.updateVector('if')

    def enterTernary_if(self, ctx):
        self.updateVector('ternary_if')

    def enterSample(self, ctx):
        self.updateVector('sample')
        lhs = ctx.expression()
        if isinstance(lhs, StanParser.Id_accessContext):
            self.sink = lhs.ID().getText()
        elif isinstance(lhs, StanParser.ArrayContext):
            self.sink = lhs.array_access().ID().getText()
        elif isinstance(lhs, StanParser.FunctionContext):
            if lhs.function_call().inbuilt().getText() == 'to_vector':
                self.sink = lhs.function_call().expression(0).getText()
            else:
                print(lhs.getText())
                return
        else:
            print(ctx.getText())
            assert False
        # else:
        #     print("Missing" + ctx.getText())

    def enterExponop(self, ctx):
        self.arithComplexity += 1
        self.updateVectorMax('arith', self.arithComplexity)
        self.updateVector('arithops')

    def exitExponop(self, ctx):
        self.arithComplexity -= 1

    def enterDivop(self, ctx):
        self.arithComplexity += 1
        self.updateVectorMax('arith', self.arithComplexity)
        self.updateVector('arithops')

    def exitDivop(self, ctx):
        self.arithComplexity -= 1

    def enterAccudivop(self, ctx):
        self.arithComplexity += 1
        self.updateVectorMax('arith', self.arithComplexity)
        self.updateVector('arithops')

    def exitAccudivop(self, ctx):
        self.arithComplexity -= 1

    def enterMulop(self, ctx):
        self.arithComplexity += 1
        self.updateVectorMax('arith', self.arithComplexity)
        self.updateVector('arithops')

    def exitMulop(self, ctx):
        self.arithComplexity -= 1

    def enterAccumulop(self, ctx):
        self.arithComplexity += 1
        self.updateVectorMax('arith', self.arithComplexity)
        self.updateVector('arithops')

    def exitAccumulop(self, ctx):
        self.arithComplexity -= 1

    def enterAddop(self, ctx):
        self.arithComplexity += 1
        self.updateVectorMax('arith', self.arithComplexity)
        self.updateVector('arithops')

    def exitAddop(self, ctx):
        self.arithComplexity -= 1

    def enterMinusop(self, ctx):
        self.arithComplexity += 1
        self.updateVectorMax('arith', self.arithComplexity)
        self.updateVector('arithops')

    def exitMinusop(self, ctx):
        self.arithComplexity -= 1


def update_table(data, output_file, keep_prefix=[], remove_prefix=[]):

    try:
        df = pd.read_csv(output_file, index_col='program')
        #print(df)
        data = pd.Series(data)


        #data["program"] = data["program"].split('/')[-1]
        # if not (data["program"] == df["program"]).any():
        # namelist = list(df)
        # namelist.remove('program')
        # order = ['program']
        # order.extend(namelist)
        # df[order].to_csv(output_file)
        newdf = pd.DataFrame(data).transpose()
        newdf.set_index('program', inplace=True)
        df = df.drop(data['program'], errors='ignore')

        df = df.append(newdf)
        #df = df.append(data)
        #df.set_index('program')
        #print(df)
        df.to_csv(output_file)
        print("Outputting...")
    except Exception as e:
        print("new file...")
        data = pd.Series(data)

        df = pd.DataFrame(data).transpose()
        df.set_index('program', inplace=True)
        for pat in remove_prefix:
            df = df.filter(set(df).difference(set(df.filter(regex=(pat)))))
        for pat in keep_prefix:
            df = df.filter((set(df.filter(regex=(pat)))))
        df.to_csv(output_file)
    # except Exception as e:
    #     print(e.message)
    #     print(e)
    #     print(data)


def autocorr(x):
    if len(x) <= 2:
        return 0.0, 0.0
    xmean = x - np.mean(x)
    xcorr = np.correlate(xmean, xmean, mode='full')
    if xcorr[len(x) - 1] != 0:
        xcorr = (xcorr+0.0)/xcorr[len(x)-1]
    else:
        return 0.0, 0.0
    # dist 1 and 2 correlations
    return xcorr[len(x)], xcorr[len(x)+1]


def isvector(shape):
    return len(shape) == 1 and shape[0] > 1


def ismatrix(shape):
    return len(shape) == 2


def getFeature(fv, feature_name):
    assert type(fv) == dict
    return 0.0 if feature_name not in fv else fv[feature_name]


def process_data(data_file, fv, observe_variables):

    try:
        if ".json" in data_file:
            with open(data_file) as data:
                jdata = json.load(data)
                for key in jdata:
                    d = jdata[key]
                    arrshape = np.shape(d)
                    if len(arrshape) > 0 and arrshape[0] > 1:
                        # vector or matrix
                        if np.all(np.isin(d, [1,0])):
                            fv['has_bool_data'] = 1
                        if np.array(d).dtype == np.int:
                            fv['has_int_data'] = 1
                        if np.array(d).dtype == np.float:
                            fv['has_float_data'] = 1
                        if np.min(d) < 0:
                            fv['has_negative_data'] = 1
                        if np.array(d).dtype == np.int and np.size(d) > 2 and np.min(d) >= -1 and np.max(d) < 10:
                            # considering -1 for missing labels in some cases
                            fv['has_categorical'] = 1
        elif ".R" in data_file:
            d=ro.r.source(data_file)
            predictor_variables = []
            for k in list(ro.r.objects()):
                d = ro.r[k]
                print(k)
                if "categories" in fv:
                    if fv["categories"] == k:
                        fv["categories"] = int(d[0])
                arrshape = np.shape(d)
                print(arrshape)
                if len(arrshape) > 0 and arrshape[0] > 1:
                    # vector or matrix data
                    if np.all(np.isin(d, [1, 0])):
                        fv['has_bool_data'] = 1
                    if np.array(d).dtype == np.int:
                        fv['has_int_data'] = 1
                    if np.array(d).dtype == np.float:
                        fv['has_float_data'] = 1
                    if np.min(d) < 0:
                        fv['has_negative_data'] = 1
                    if np.array(d).dtype == np.int and np.size(d) > 2 and np.min(d) >= -1 and np.max(d) < 10:
                        # considering -1 for missing labels in some cases
                        fv['has_categorical'] = 1

                    x = np.array(d).flatten()
                    if k in observe_variables:
                        sparsity_observe = getFeature(fv, 'dt_sparsity_observe')
                        sparsity_observe = max(((sum(x == 0) + 0.0) / len(x)), sparsity_observe)
                        try:
                            fv['dt_sparsity_observe'] = sparsity_observe
                            fv['dt_observe_mean'] = max(getFeature(fv, 'dt_observe_mean'), np.mean(x))
                            fv['dt_observe_std'] = max(getFeature(fv, 'dt_observe_std'), np.std(x))
                            fv['dt_observe_coeff_var'] = max(getFeature(fv, 'dt_observe_coeff_var'), np.std(x) / np.mean(x))
                            if len(x) > 8:
                                fv['dt_observe_skew'] = max(getFeature(fv,'dt_observe_skew'), skewtest(x, axis=None, nan_policy='omit')[0])
                            if len(x) >= 20:
                                fv['dt_observe_kurtosis'] = max(getFeature(fv,'dt_observe_kurtosis'), kurtosistest(x, axis=None, nan_policy='omit')[0])
                        except Exception as e:
                            traceback.print_exc(e)
                    else:
                        predictor_variables.append(k)
                        try:
                            sparsity_predictor = getFeature(fv, 'dt_sparsity_predictor')
                            sparsity_predictor = max(((sum(x == 0) + 0.0) / len(x)), sparsity_predictor)
                            fv['dt_sparsity_predictor'] = sparsity_predictor
                            if len(x) > 8:
                                fv['dt_predictor_skew'] = max(getFeature(fv, 'dt_predictor_skew'), skewtest(x,axis=None, nan_policy='omit')[0])
                            if len(x) >= 20:
                                fv['dt_predictor_kurtosis'] = max(getFeature(fv, 'dt_predictor_kurtosis'),kurtosistest(x, axis=None, nan_policy='omit')[0])
                        except Exception as e:
                            traceback.print_exc(e)

                # compute correlation coeff in columns of predictor variables
                if len(arrshape) > 1 and k not in observe_variables:
                    # for matrices
                    maxkt = getFeature(fv, 'dt_max_kt')
                    for c in range(arrshape[1] - 1):
                        x1 = np.array(d)[:, c]

                        x2 = np.array(d)[:, c+1]
                        kt = kendalltau(x1, x2)[0]
                        maxkt = max(maxkt, abs(kt))
                    fv['dt_max_kt'] = maxkt

                if len(arrshape) == 1 and arrshape[0] > 1 and k in observe_variables:
                    # autocorrelation in observed variable(s), only vectors
                    try:
                        max_autocorr_observe_1 = getFeature(fv, 'dt_max_autocorr_observe_1')
                        max_autocorr_observe_2 = getFeature(fv, 'dt_max_autocorr_observe_2')
                        var_autocorr = autocorr(ro.r[k])
                        fv['dt_max_autocorr_observe_1'] = max(max_autocorr_observe_1, var_autocorr[0])
                        fv['dt_max_autocorr_observe_2'] = max(max_autocorr_observe_2, var_autocorr[1])
                    except Exception as e:
                        traceback.print_exc(e)
                elif len(arrshape) == 1 and arrshape[0] > 1:
                    # autocorrelation in observed variable(s), only for vectors
                    try:
                        max_autocorr_predictor_1 = getFeature(fv, 'dt_max_autocorr_predictor_1')
                        max_autocorr_predictor_2 = getFeature(fv, 'dt_max_autocorr_predictor_2')
                        var_autocorr = autocorr(ro.r[k])
                        fv['dt_max_autocorr_predictor_1'] = max(max_autocorr_predictor_1, var_autocorr[0])
                        fv['dt_max_autocorr_predictor_2'] = max(max_autocorr_predictor_2, var_autocorr[1])
                    except Exception as e:
                        traceback.print_exc(e)

            # correlation among predictor variables
            max_pred_kt = getFeature(fv, 'dt_max_pred_kt')
            for i in range(len(predictor_variables)-1):
                var1 = np.array(ro.r[predictor_variables[i]]).flatten()
                var2 = np.array(ro.r[predictor_variables[i+1]]).flatten()
                if len(var1) != len(var2):
                    continue
                max_pred_kt = max(max_pred_kt, kendalltau(var1,var2)[0])
            fv['dt_max_pred_kt'] = max_pred_kt

            # correlation between predictor and observe variables
            if len(observe_variables) > 0:
                max_predictor_observe_corr =  getFeature(fv, 'dt_max_predictor_observe_corr')
                observe_variable = np.array(ro.r[observe_variables[0]])
                for predictor_variable in predictor_variables:
                    var = np.array(ro.r[predictor_variable])
                    if isvector(observe_variable) and isvector(var) and len(observe_variable) == len(var):
                        #print('vector and lengths match')
                        max_predictor_observe_corr = max(max_predictor_observe_corr, kendalltau(var, observe_variable)[0])
                    elif isvector(observe_variable) and ismatrix(var) and len(observe_variable) == len(var): # rows equal
                        # compute row means
                        #print('computing row means for predictor')
                        max_predictor_observe_corr = max(max_predictor_observe_corr, kendalltau(np.mean(var, axis=1), observe_variable)[0])

                fv['dt_max_predictor_observe_corr'] = max_predictor_observe_corr

        fv['data_size'] = os.path.getsize(data_file)
    except Exception as e:
        traceback.print_exc(e)
        fv['data_size'] = os.path.getsize(data_file)
        pass


if __name__ == "__main__":
    data_file = sys.argv[1]
    fv_jsonFile = open(sys.argv[2])
    fv_jsonString = fv_jsonFile.read()
    fv = json.loads(fv_jsonString)
    obv = []
    obv_filepath = sys.argv[3]
    with open(obv_filepath) as fp:
        obv_string = fp.read()
        obv = obv_string.split("\n")
    process_data(data_file, fv, obv)
    f = open("/Users/zhekunz2/Desktop/SixthSense/c4pp/output/test.txt", "a+")
    f.write(data_file+"\n")

    # stan_file_path=sys.argv[1]
    # results_file=sys.argv[2]
    # data_file=sys.argv[3]
    # try:
    #     output_file=sys.argv[4]
    # except:
    #     output_file="anyhow_this_file_doesnt_exist"
    #
    # try:
    #     # template file
    #     templatefile = sys.argv[5]
    #     print("Computing Graph features...")
    #
    #     modelBuilder = ModelBuilder(templatefile)
    #     graph_features = modelBuilder.get_features()
    #     print("Done Graph features...")
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc(e)
    #     graph_features = None
    #
    # try:
    #     pyrofile = sys.argv[6]
    #     print(pyrofile)
    #     print("Computing Pyro features...")
    #
    #     pyrofeatureparser = PyroFeatureParser()
    #     pyro_features = pyrofeatureparser.get_pyro_features(pyrofile)
    #     print("Done Pyro features...")
    # except Exception as e:
    #     pass
    #     #print(e)
    #     #print("Error pyro")
    #
    #     pyro_features = None
    # try:
    #     stanfile = antlr4.FileStream(stan_file_path)
    #     lexer = StanLexer(stanfile)
    #     stream = antlr4.CommonTokenStream(lexer)
    #     parser = StanParser(stream)
    #     featureParser = StanFeatureParser()
    #     code = parser.program()
    #     walker = ParseTreeWalker()
    #     walker.walk(featureParser, code)
    #     fv = featureParser.feature_vector
    #
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc(e)
    #     print("cant open stanfile")
    #     fv = {}
    # print(fv)
    # print(featureParser.observe_variables)
    # process_data(data_file, fv, featureParser.observe_variables if featureParser is not None else None)
    #
    # fv['program'] = stan_file_path
    #
    # program_name=stan_file_path.split('/')[-2]
    #
    # try:
    #     results = open(results_file).read()
    #     results = re.findall(program_name+"[^\n]*", results)[0]
    #     result_strip = results.strip().split(",")
    #     if len(result_strip) <= 4:
    #         fv['results'] = result_strip[2]
    #         fv['value'] = result_strip[3]
    #     elif len(result_strip) >= 12:
    #         fv['t_result'] = result_strip[2]
    #         fv['t_value'] = result_strip[3]
    #         fv['ks_result'] = result_strip[4]
    #         fv['ks_value'] = result_strip[5]
    #         fv['kl_result'] = result_strip[6]
    #         fv['kl_value'] = result_strip[7]
    #         fv['smkl_result'] = result_strip[8]
    #         fv['smkl_value'] = result_strip[9]
    #         fv['hell_result'] = result_strip[10]
    #         fv['hell_value'] = result_strip[11]
    #         if len(result_strip) >= 14:
    #             fv['rhat1_result'] = result_strip[12]
    #             fv['rhat1_value'] = result_strip[13]
    #         if len(result_strip) >= 16:
    #             fv['rhat2_result'] = result_strip[14]
    #             fv['rhat2_value'] = result_strip[15]
    #     else:
    #         fv['results'] = "Error"
    #         fv['value'] = "NA"
    #
    # except:
    #     pass
    #
    # # graph features
    # if graph_features is not None:
    #     for key in graph_features:
    #         fv[key] = graph_features[key]
    #
    # if pyro_features is not None:
    #     for key in pyro_features:
    #         fv[key] = pyro_features[key]
    # #print(fv)
    # print("Feature Vector Done...")
    # update_table(fv, output_file)




#!/usr/bin/env python
# python feature_ast.py [template_file] [output_file] [parent level]
from nearpy.hashes import RandomDiscretizedProjections

from parser.template.Template2Lexer import Template2Lexer
from parser.template.Template2Listener import Template2Listener
from parser.template.Template2Parser import Template2Parser
import pandas as pd
import glob, os
import hashlib

fixed_hashes = {'TernaryContext': 58, 'DataContext': 17, 'ParamsContext': 48, 'Function_declContext': 32, 'VecdivopContext': 64, 'StringContext': 55, 'Array_accessContext': 13, 'BracketsContext': 16, 'NeContext': 44, 'GtContext': 36, 'TemplateContext': 57, 'DimsContext': 19, 'LimitsContext': 39, 'VecmulopContext': 65, 'MulopContext': 43, 'LtContext': 41, 'AddopContext': 10, 'MinusopContext': 42, 'Return_or_param_typeContext': 53, 'DivopContext': 21, 'SubsetContext': 56, 'RefContext': 52, 'Function_callContext': 31, 'If_stmtContext': 37, 'FunctionContext': 30, 'TransformeddataContext': 59, 'ObserveContext': 46, 'PriorContext': 50, 'EqContext': 24, 'NumberContext': 45, 'ArrayContext': 12, 'DtypeContext': 22, 'FparamContext': 28, 'ParamContext': 47, 'LeqContext': 38, 'VectorDIMSContext': 67, 'TransformedparamContext': 60, 'QueryContext': 51, 'LoopcompContext': 40, 'BlockContext': 15, 'AssignContext': 14, 'FunctionsContext': 33, '__class__': 68, 'UnaryContext': 62, 'FparamsContext': 29, 'ExprContext': 26, 'DistexprContext': 20, 'StatementContext': 54, 'AndContext': 11, 'GeneratedquantitiesContext': 34, 'For_loopContext': 27, 'TransposeContext': 61, 'DeclContext': 18, 'ValContext': 63, 'ExponopContext': 25, 'PrimitiveContext': 49, 'GeqContext': 35, 'Else_blkContext': 23, 'VectorContext': 66}

class FeatureBuilder(Template2Listener):

    def __init__(self, level):
        self.feature_vector = {}
        self.parentLevel = int(level)
        print(self.parentLevel)
        self.hashes = dict()

        self.rdp = RandomDiscretizedProjections('rdp', 5, 6, rand_seed=98412194)
        self.rdp.reset(self.parentLevel)

    def getHash(self, vector):
        if len(vector) < self.parentLevel:
            vector = vector + (self.parentLevel - len(vector)) * [0]
        h = self.rdp.hash_vector(vector)[0]
        # h = '_'.join([str(x) for x in vector])

        return h

    def getParents(self, ctx):
        curLevel = 0
        curNode = ctx
        path = []
        while curNode is not None and curLevel < self.parentLevel:
            #path.append(curNode.getRuleIndex())
            nodename = curNode.__class__.__name__
            path.append(fixed_hashes[nodename])
            curLevel += 1
            curNode = curNode.parentCtx
        return path

    def update_vector(self, ctx):
        if self.parentLevel <= 1:
            name = type(ctx).__name__
            if ctx.parentCtx is not None:
                parentName = type(ctx.parentCtx).__name__
                feature_name = 't_' + parentName + '_' + name
                if feature_name not in self.feature_vector:
                    self.feature_vector[feature_name] = 0
                self.feature_vector[feature_name] += 1
        else:
            path=self.getParents(ctx)
            name=self.getHash(path)

        if name not in self.feature_vector:
            self.feature_vector[name] = 0
        self.feature_vector[name] += 1

    def enterAddop(self, ctx):
        self.update_vector(ctx)

    def enterAnd(self, ctx):
        self.update_vector(ctx)

    def enterArray(self, ctx):
        self.update_vector(ctx)

    def enterArray_access(self, ctx):
        self.update_vector(ctx)

    def enterAssign(self, ctx):
        self.update_vector(ctx)

    def enterBlock(self, ctx):
        self.update_vector(ctx)

    def enterBrackets(self, ctx):
        self.update_vector(ctx)

    def enterData(self, ctx):
        self.update_vector(ctx)

    def enterDecl(self, ctx):
        self.update_vector(ctx)

    def enterPrimitive(self, ctx):
        self.update_vector(ctx)

    def enterNumber(self, ctx):
        self.update_vector(ctx)

    def enterDtype(self, ctx):
        self.update_vector(ctx)

    def enterVector(self, ctx):
        self.update_vector(ctx)

    def enterDims(self, ctx):
        self.update_vector(ctx)

    def enterVectorDIMS(self, ctx):
        self.update_vector(ctx)

    def enterLimits(self, ctx):
        self.update_vector(ctx)

    def enterPrior(self, ctx):
        self.update_vector(ctx)

    def enterParam(self, ctx):
        self.update_vector(ctx)

    def enterParams(self, ctx):
        self.update_vector(ctx)

    def enterDistexpr(self, ctx):
        self.update_vector(ctx)

    def enterLoopcomp(self, ctx):
        self.update_vector(ctx)

    def enterFor_loop(self, ctx):
        self.update_vector(ctx)

    def enterIf_stmt(self, ctx):
        self.update_vector(ctx)

    def enterElse_blk(self, ctx):
        self.update_vector(ctx)

    def enterFunction_call(self, ctx):
        self.update_vector(ctx)

    def enterFparam(self, ctx):
        self.update_vector(ctx)

    def enterFparams(self, ctx):
        self.update_vector(ctx)

    def enterReturn_or_param_type(self, ctx):
        self.update_vector(ctx)

    def enterFunction_decl(self, ctx):
        self.update_vector(ctx)

    def enterTransformedparam(self, ctx):
        self.update_vector(ctx)

    def enterTransformeddata(self, ctx):
        self.update_vector(ctx)

    def enterGeneratedquantities(self, ctx):
        self.update_vector(ctx)

    def enterFunctions(self, ctx):
        self.update_vector(ctx)

    def enterVal(self, ctx):
        self.update_vector(ctx)

    def enterDivop(self, ctx):
        self.update_vector(ctx)

    def enterString(self, ctx):
        self.update_vector(ctx)

    def enterExponop(self, ctx):
        self.update_vector(ctx)

    def enterMinusop(self, ctx):
        self.update_vector(ctx)

    def enterLt(self, ctx):
        self.update_vector(ctx)

    def enterUnary(self, ctx):
        self.update_vector(ctx)

    def enterEq(self, ctx):
        self.update_vector(ctx)

    def enterGt(self, ctx):
        self.update_vector(ctx)

    def enterRef(self, ctx):
        self.update_vector(ctx)

    def enterGeq(self, ctx):
        self.update_vector(ctx)

    def enterMulop(self, ctx):
        self.update_vector(ctx)

    def enterFunction(self, ctx):
        self.update_vector(ctx)

    def enterVecmulop(self, ctx):
        self.update_vector(ctx)

    def enterNe(self, ctx):
        self.update_vector(ctx)

    def enterLeq(self, ctx):
        self.update_vector(ctx)

    def enterTranspose(self, ctx):
        self.update_vector(ctx)

    def enterVecdivop(self, ctx):
        self.update_vector(ctx)

    def enterTernary(self, ctx):
        self.update_vector(ctx)

    def enterSubset(self, ctx):
        self.update_vector(ctx)

    def enterObserve(self, ctx):
        self.update_vector(ctx)

    def enterStatement(self, ctx):
        self.update_vector(ctx)

    def enterQuery(self, ctx):
        self.update_vector(ctx)

    def enterTemplate(self, ctx):
        self.update_vector(ctx)


def update_table(data, output_file):
    try:
        df = pd.read_csv(output_file, index_col='program')
        data = pd.Series(data)
        newdf = pd.DataFrame(data).transpose()
        print(newdf)
        newdf.set_index('program', inplace=True)
        df = df.drop(data['program'], errors='ignore')
        df = df.append(newdf)
        df.to_csv(output_file)
        print("Outputting...")
    except Exception as e:
        print("new file...")
        data = pd.Series(data)
        df = pd.DataFrame(data).transpose()
        df.set_index('program', inplace=True)
        df.to_csv(output_file)


if __name__ == '__main__':
    import antlr4
    from antlr4 import *
    import sys
    import time
    start_time = time.time()
    try:
        templatefile = antlr4.FileStream(sys.argv[1])
        lexer = Template2Lexer(templatefile)
        stream = antlr4.CommonTokenStream(lexer)
        parser = Template2Parser(stream)
        featureParser = FeatureBuilder(sys.argv[3] if len(sys.argv) > 3 else 2)
        code = parser.template()
        walker = ParseTreeWalker()
        walker.walk(featureParser, code)
        fv = featureParser.feature_vector
        stanfile=glob.glob(sys.argv[1].replace("ctemplate.template", "*.stan"))[0]
        print(stanfile)
        fv['program'] = stanfile
        print(fv)
        print("size of fv: "+str(len(fv)))
        update_table(fv, sys.argv[2])
    except Exception as e:
        print("error")
        import traceback
        traceback.print_exc(e)
    print("--- %s seconds ---" % (time.time() - start_time))
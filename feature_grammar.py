#!/usr/bin/env python
# feature_grammar.py [template_file] [outputfile] [parent_depth]
from parser.template.Template2Lexer import Template2Lexer
from parser.template.Template2Listener import Template2Listener
from parser.template.Template2Parser import Template2Parser
import pandas as pd
import glob, os

class FeatureBuilder(Template2Listener):

    def __init__(self):
        self.feature_vector = {}

    def update_vector(self, ctx):
        name = type(ctx).__name__
        if ctx.parentCtx is not None:
            parentName = type(ctx.parentCtx).__name__
            feature_name = 't_' + parentName + '_' + name
            if feature_name not in self.feature_vector:
                self.feature_vector[feature_name] = 0
            self.feature_vector[feature_name] += 1

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

    try:
        templatefile = antlr4.FileStream(sys.argv[1])
        lexer = Template2Lexer(templatefile)
        stream = antlr4.CommonTokenStream(lexer)
        parser = Template2Parser(stream)
        featureParser = FeatureBuilder()
        code = parser.template()
        walker = ParseTreeWalker()
        walker.walk(featureParser, code)
        fv = featureParser.feature_vector
        stanfile=glob.glob(sys.argv[1].replace("ctemplate.template", "*.stan"))[0]
        print(stanfile)
        fv['program'] = stanfile
        update_table(fv, sys.argv[2])
    except Exception as e:
        print("error")
        import traceback
        traceback.print_exc(e)
# Generated from Template2.g4 by ANTLR 4.7.1
from antlr4 import *

# This class defines a complete generic visitor for a parse tree produced by Template2Parser.

class Template2Visitor(ParseTreeVisitor):

    # Visit a parse tree produced by Template2Parser#primitive.
    def visitPrimitive(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#number.
    def visitNumber(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#dtype.
    def visitDtype(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#array.
    def visitArray(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#vector.
    def visitVector(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#dims.
    def visitDims(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#vectorDIMS.
    def visitVectorDIMS(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#decl.
    def visitDecl(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#limits.
    def visitLimits(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#data.
    def visitData(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#prior.
    def visitPrior(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#param.
    def visitParam(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#params.
    def visitParams(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#distexpr.
    def visitDistexpr(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#loopcomp.
    def visitLoopcomp(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#for_loop.
    def visitFor_loop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#if_stmt.
    def visitIf_stmt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#else_blk.
    def visitElse_blk(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#function_call.
    def visitFunction_call(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#fparam.
    def visitFparam(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#fparams.
    def visitFparams(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#return_or_param_type.
    def visitReturn_or_param_type(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#function_decl.
    def visitFunction_decl(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#block.
    def visitBlock(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#transformedparam.
    def visitTransformedparam(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#transformeddata.
    def visitTransformeddata(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#generatedquantities.
    def visitGeneratedquantities(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#functions.
    def visitFunctions(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#val.
    def visitVal(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#divop.
    def visitDivop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#string.
    def visitString(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#exponop.
    def visitExponop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#array_access.
    def visitArray_access(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#addop.
    def visitAddop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#minusop.
    def visitMinusop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#lt.
    def visitLt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#unary.
    def visitUnary(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#eq.
    def visitEq(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#gt.
    def visitGt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#brackets.
    def visitBrackets(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#ref.
    def visitRef(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#geq.
    def visitGeq(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#mulop.
    def visitMulop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#and.
    def visitAnd(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#function.
    def visitFunction(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#vecmulop.
    def visitVecmulop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#ne.
    def visitNe(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#leq.
    def visitLeq(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#transpose.
    def visitTranspose(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#vecdivop.
    def visitVecdivop(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#ternary.
    def visitTernary(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#subset.
    def visitSubset(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#assign.
    def visitAssign(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#observe.
    def visitObserve(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#statement.
    def visitStatement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#query.
    def visitQuery(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Template2Parser#template.
    def visitTemplate(self, ctx):
        return self.visitChildren(ctx)



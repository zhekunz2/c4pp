#!/usr/bin/env python

import sys
import hashlib
import antlr4
from antlr4 import ParseTreeWalker, ParserRuleContext

from template.Template2Lexer import Template2Lexer
from template.Template2Listener import Template2Listener
from template.Template2Parser import Template2Parser
from nearpy.hashes import RandomBinaryProjections,RandomDiscretizedProjections


MAX_PATH_LENGTH = 100

stacks = dict()
#rdp = RandomBinaryProjections('rbp', 100, rand_seed=98412194)
rdp = RandomDiscretizedProjections('rdp', 10, 1000, rand_seed=98412194)
rdp.reset(MAX_PATH_LENGTH)


def getHash(vector):
    # if len(vector) < MAX_PATH_LENGTH:
    #     vector = vector + (MAX_PATH_LENGTH-len(vector))*[0]
    h=rdp.hash_vector(vector)[0]

    return h


def update(d, entry):
    if entry in d:
        d[entry] += 1
    else:
        d[entry] = 1


# collect leaves
class LeavesCollector(Template2Listener):
    def __init__(self, file):
        self.lf = set()
        self.mappings = dict()
        self.data = []
        self.priors = []
        self.locals = []
        self.enabled = True
        self.section = 'model'
        stanfile = antlr4.FileStream(file)
        lexer = Template2Lexer(stanfile)
        stream = antlr4.CommonTokenStream(lexer)
        parser = Template2Parser(stream)
        self.code = parser.template()
        walker = ParseTreeWalker()
        walker.walk(self, self.code)
        self.getLeaves()

    def getLeaves(self):
        nodes = [self.code]
        while len(nodes) != 0:
            newnodes = []
            for n in nodes:
                if isinstance(n, ParserRuleContext) and len(n.children)>0:
                    if isinstance(n, Template2Parser.DataContext):
                        continue
                    if isinstance(n, Template2Parser.GeneratedquantitiesContext):
                        continue
                    newnodes = newnodes + n.children
                else:
                    if isinstance(n, antlr4.TerminalNode):
                        if n.symbol.type in [Template2Parser.ID, Template2Parser.FLOATTYPE,
                                             Template2Parser.INTEGERTYPE, Template2Parser.INT,
                                             Template2Parser.DOUBLE,Template2Parser.COMPLEX,
                                             Template2Parser.FUNCTION]:

                            if n.symbol.type == Template2Parser.ID:
                                id=n.getText()
                                if id in self.priors:
                                    self.mappings[n] = "prior"
                                    self.lf.add(n)
                                elif id in self.data:
                                    self.mappings[n] = "data"
                                    self.lf.add(n)
                                elif id in self.locals:
                                    self.mappings[n] = "local"
                                    self.lf.add(n)
                                elif isinstance(n.parentCtx, Template2Parser.For_loopContext):
                                    self.mappings[n] = "local"
                                    self.lf.add(n)
                                # else:
                                #     print('missing ', id)
                            else:
                                self.mappings[n] = n.getText()

            nodes = newnodes

    def enterData(self, ctx):
        self.enabled = False
        self.data.append(ctx.ID().getText())

    def exitData(self, ctx):
        self.enabled = True

    def enterTransformeddata(self, ctx):
        self.section = 'transformeddata'

    def exitTransformeddata(self, ctx):
        self.section = 'model'

    def enterTransformedparam(self, ctx):
        self.section = 'transformedparam'

    def exitTransformedparam(self, ctx):
        self.section = 'model'

    def enterPrior(self, ctx):
        self.priors.append(ctx.expr().ID().getText())

    def enterDecl(self, ctx):
        if self.section == 'transformedparam':
            self.priors.append(ctx.ID().getText())
        else:
            self.locals.append(ctx.ID().getText())


    def enterGeneratedquantities(self, ctx):
        self.enabled = False

    def exitGeneratedquantities(self, ctx):
        self.enabled = True

    # def enterVal(self, ctx):
    #     if self.enabled:
    #         self.leaves.append(ctx)
    #
    # def enterString(self, ctx):
    #     if self.enabled:
    #         self.leaves.append(ctx)
    #
    # def enterRef(self, ctx):
    #     if self.enabled:
    #         self.leaves.append(ctx)
    #
    # def enterArray_access(self, ctx):
    #     if self.enabled:
    #         self.leaves.append(ctx)
    #
    # def enterDtype(self, ctx):
    #     if self.enabled:
    #         self.leaves.append(ctx)



def getTreeStack(original_ctx):
    if original_ctx in stacks:
        return stacks[original_ctx]
    stack = []
    ctx=original_ctx.parentCtx
    while ctx is not None:
        stack.append(ctx)
        ctx=ctx.parentCtx
    stacks[original_ctx]=stack
    return stack


def getPath(s, t):
    source=getTreeStack(s)
    target=getTreeStack(t)
    sourcei = len(source)-1
    targeti = len(target)-1
    path = []
    while sourcei >= 0 and targeti >= 0 and source[sourcei] == target[targeti]:
        sourcei-=1
        targeti-=1

    for i in range(sourcei+1):
        path.append((source[i].getRuleIndex()+100))

    path.append((source[sourcei+1].getRuleIndex()))

    for i in range(targeti, -1, -1):
        path.append((target[i].getRuleIndex()-100))

    return path


# def leafToHash(lf, leavesCollector):
#     if lf , Template2Parser.ID)
    # if isinstance(leaf, Template2Parser.RefContext):
    #     id=leaf.ID().getText()
    #     if id in leavesCollector.priors:
    #         return "prior"
    #     elif id in leavesCollector.data:
    #         return "data"
    #     else:
    #         return "local"
    # elif isinstance(leaf,Template2Parser.Array_accessContext):
    #     id = leaf.ID().getText()
    #     if id in leavesCollector.priors:
    #         return "prior"
    #     elif id in leavesCollector.data:
    #         return "data"
    #     else:
    #         return "local"
    # elif isinstance(leaf, Template2Parser.STRING):
    #     return "STRING"
    # elif isinstance(leaf, Template2Parser.ValContext):
    #     return leaf.getText()



def getHashes(file):
    leavesCollector = LeavesCollector(file)
    leaves = list(leavesCollector.lf)

    # print(leavesCollector.lf)
    # print(leavesCollector.priors)
    # print(leavesCollector.data)


    paths = []
    hashes = dict()
    #maxlength = 0

    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            path = getPath(leaves[i], leaves[j])

            if len(path) > (MAX_PATH_LENGTH - 2):
                continue

            path = path + [0]*(MAX_PATH_LENGTH-2-len(path))
            path = [abs(hash(leavesCollector.mappings[leaves[i]])) % (10 ** 8)] + path + \
                   [abs(hash(leavesCollector.mappings[leaves[j]])) % (10 ** 8)]
            update(hashes, getHash(path))
    return hashes


hashes=getHashes(sys.argv[1])
keys=hashes.keys()
#print('program,'+','.join(keys))
#print(sys.argv[1]+ ',' + ','.join([str(hashes[x]) for x in keys]))
hashes['program'] = sys.argv[1]
#print(hashes)
print(len(hashes.keys()))

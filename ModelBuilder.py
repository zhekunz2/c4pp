import antlr4
from antlr4 import *

from parser.template.Template2Visitor import Template2Visitor
from parser.template.Template2Lexer import Template2Lexer
from parser.template.Template2Parser import Template2Parser
from parser.template.Template2Listener import Template2Listener
import networkx as nx
from networkx.algorithms import approximation as alg
import numpy as np

class Analyzer(Template2Listener):
    def __init__(self, templatefile):

        self.track_enable = False
        self.templatefile=templatefile
        self.dep_map = {}
        self.cur_var = None
        self.cur_parent = None
        self.priors = set()
        self.intransformedparam = False
        self.data = []
        self.data_constants = []
        self.target = []
        self.array_outer = []
        self.array_inner = []
        self.find_vars()

    def getparser(self,templatefile):
        template = antlr4.FileStream(templatefile)
        lexer = Template2Lexer(template)
        stream = antlr4.CommonTokenStream(lexer)
        parser = Template2Parser(stream)
        return parser

    def getDeps(self, var):
        deps = []

        deps= deps + self.dep_map.get(var, [])
        deps = list(set(deps))
        removed = []
        while True:
            deps2 = []
            for d in deps:
                if d in self.priors:
                    deps2.append(d)
                elif d in self.data and d not in self.data_constants:
                    deps2.append(d)
                else:
                    deps2 = deps2 + self.dep_map.get(d,[])

            deps2 = list(set(deps2))
            deps2 = filter(lambda x: x not in removed, deps2)
            if set(deps2) == set(deps):
                break
            else:
                removed = removed + list(set(deps).difference(set(deps2)))
                #print("Remove: ", removed)
            deps = deps2
        #print(deps)
        return deps

    def find_vars(self):
        parser = self.getparser(self.templatefile)
        walker = ParseTreeWalker()
        walker.walk(self, parser.template())
        # for a in self.dep_map:
        #     print("{0} : {1}".format(a, self.dep_map[a]))
        # print("Priors : {0}".format(self.priors))

    def enterData(self, ctx):
        self.data.append(ctx.ID().getText())
        if ctx.expr() is not None:
            self.data_constants.append(ctx.ID().getText())


    def enterAssign(self, ctx):
        id = ctx.ID().getText() if ctx.ID() is not None else ctx.expr(0).ID().getText()
        self.cur_var = id
        self.track_enable = True
        self.cur_parent = ctx
        if self.intransformedparam:
            self.priors.add(id)
        elif id == 'target':
            self.target.append(id)

    def exitAssign(self, ctx):
        self.track_enable = False

    def enterObserve(self, ctx):
        #id = ctx.expr(1).ID().getText() if len(ctx.expr()) > 1 else ctx.expr(0).ID().getText()
        id = ctx.expr(1).ID().getText() if len(ctx.expr()) > 1 else ctx.expr(0).ID().getText()
        self.cur_var = id
        self.track_enable = True
        self.cur_parent = ctx

    def exitObserve(self, ctx):
        self.track_enable = False
        self.cur_var = None

    def enterPrior(self, ctx):
        id = ctx.expr().ID().getText()
        self.cur_var = id
        self.track_enable = True
        self.cur_parent = ctx
        self.priors.add(id)

    def exitPrior(self, ctx):
        self.track_enable = False
        self.cur_var = None

    def enterTransformedparam(self, ctx):
        self.intransformedparam = True

    def exitTransformedparam(self, ctx):
        self.intransformedparam = False

    def enterRef(self, ctx):
        if self.track_enable and self.cur_var is not None:
            v = ctx.getText()
            if v==self.cur_var:
                if not self.checkside(self.cur_parent, ctx):
                    self.update_var(self.cur_var, v)
                    #print("Adding " + v + " ....")
                #else:
                #    print("Skipping" + v + " ....")
            else:
                self.update_var(self.cur_var, v)

        if self.array_outer is not None and len(self.array_outer) > 0:
            self.update_var(self.array_outer[-1], ctx.getText())
                #print("Adding " + v + " ....")

    def enterArray_access(self, ctx):
        if self.track_enable and self.cur_var is not None:
            v = ctx.ID().getText()
            if v == self.cur_var:
                if not self.checkside(self.cur_parent, ctx):
                    self.update_var(self.cur_var, v)
                    #print("Adding " + v + " ....")
                #else:
                #     print("Skipping" + v + " ....")
            else:
                self.update_var(self.cur_var, v)

        # tracking array references
        if self.array_outer is not None and len(self.array_outer) > 0:
            self.update_var(self.array_outer[-1], ctx.ID().getText())

        self.array_outer.append(ctx.ID().getText())

    def exitArray_access(self, ctx):
        self.array_outer.pop()

    def update_var(self, var, dep):
        if var not in self.dep_map:
            self.dep_map[var] = [dep]
        else:
            self.dep_map[var].append(dep)

    def checkside(self, parentCtx, ctx):
        c = ctx
        while c is not None and c.parentCtx != parentCtx:
            c = c.parentCtx
        if isinstance(parentCtx, Template2Parser.ObserveContext) and parentCtx.children[1] == c:
            return True
        if parentCtx.children[0] == c:
            #print(c)
            return True
        return False

    def enterFunction_call(self, ctx=Template2Parser.Function_callContext):
        if ctx.FUNCTION().getText() == 'increment_log_prob':
            self.track_enable = True
            self.cur_var = 'target'
            if len(self.target) == 0:
                self.target.append('target')

    def exitFunction_call(self, ctx):
        if ctx.FUNCTION().getText() == 'increment_log_prob':
            self.track_enable = False
            self.cur_var = None

class Node:
    def __init__(self, name):
        self.name = name
        self.density = None
        self.succ = []
        self.is_observed = False
        self.observedVal = None

class ModelBuilder(Template2Listener):
    def __init__(self,templatefile):
        self.graph = nx.DiGraph()
        self.templatefile = templatefile
        self.analyzer = Analyzer(self.templatefile)
        template = antlr4.FileStream(templatefile)
        lexer = Template2Lexer(template)
        stream = antlr4.CommonTokenStream(lexer)
        parser = Template2Parser(stream)
        template = parser.template()
        walker = ParseTreeWalker()
        walker.walk(self, template)


    def enterObserve(self, ctx):
        if len(ctx.expr()) > 1:
            node = Node(ctx.expr(1).ID().getText())
        else:
            node = Node(ctx.expr(0).ID().getText())
        self.graph.add_node(node.name, data=node, obs=True)
        #print(nx.get_node_attributes(self.graph, node))

    def enterPrior(self, ctx):
        node = Node(ctx.expr().ID().getText())
        self.graph.add_node(node.name, data=node)

    def exitTemplate(self, ctx):
        for n in self.analyzer.priors:
            self.graph.add_node(n)
        for x in self.analyzer.target:
            self.graph.add_node(x)

        while True:
            curnodes = [x for x in self.graph.nodes]
            for n in curnodes:
                deps = self.analyzer.getDeps(n)

                #print('Deps: {0} : {1}'.format(n, deps))
                if deps is not None:
                    for d in deps:
                        if n in self.analyzer.data:
                            if d in self.analyzer.data:
                                continue
                            self.graph.add_edge(n, d)
                        else:
                            self.graph.add_edge(d, n)
            if len(curnodes) == len(self.graph.nodes):
                break


    def showGraph(self):
        import matplotlib.pyplot as plt
        colors = ['green' if x in self.analyzer.data else 'orange' for x in self.graph.nodes]
        pos = nx.fruchterman_reingold_layout(self.graph)
        #nx.draw(self.graph, pos)
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors)
        nx.draw_networkx_labels(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos)
        #nx.nx_pydot.write_dot(self.graph, "graph.dot")
        from networkx.drawing.nx_agraph import to_agraph
        A =to_agraph(self.graph)
        A.layout('dot')
        A.draw('myg.png')
        #nx.draw(self.graph, with_labels=True)


        plt.show()

    def get_features(self):
        feature_vector={}
        feature_vector['g_nodes'] = len(self.graph.nodes.keys())
        feature_vector['g_edges'] = len(self.graph.edges.keys())
        feature_vector['g_selfloops'] = len(list(nx.nodes_with_selfloops(self.graph)))
        feature_vector['g_max_degree'] = max([d for n,d in self.graph.degree()])
        feature_vector['g_mean_degree'] = np.mean([d for n,d in self.graph.degree()])
        feature_vector['g_sd_degree'] = np.std([d for n,d in self.graph.degree()])

        feature_vector['g_max_in_degree'] = max([d for n, d in self.graph.in_degree()])
        feature_vector['g_mean_in_degree'] = np.mean([d for n, d in self.graph.in_degree()])
        feature_vector['g_sd_in_degree'] = np.std([d for n, d in self.graph.in_degree()])

        feature_vector['g_max_out_degree'] = max([d for n, d in self.graph.out_degree()])
        feature_vector['g_mean_out_degree'] = np.mean([d for n, d in self.graph.out_degree()])
        feature_vector['g_sd_out_degree'] = np.std([d for n, d in self.graph.out_degree()])

        feature_vector['g_density'] = nx.density(self.graph)
        feature_vector['g_independent'] = sum([1 if d == 0 else 0 for n,d in self.graph.in_degree()])
        feature_vector['g_dependent'] = sum([1 if d != 0 else 0 for n,d in self.graph.in_degree()])
        feature_vector['g_leaves'] = sum([1 if d == 0 else 0 for n,d in self.graph.out_degree()])

        scc = filter(lambda x: len(x) > 1,list(nx.strongly_connected_components(self.graph)))
        feature_vector['g_scc'] = len(scc)
        feature_vector['g_succ_size_max'] = max([len(x) for x in scc]) if len(scc) > 0 else 0
        feature_vector['g_succ_size_mean'] = np.mean([len(x) for x in scc]) if len(scc) > 0 else 0
        feature_vector['g_succ_size_sd'] = np.std([len(x) for x in scc]) if len(scc) > 0 else 0

        return feature_vector


if __name__ == '__main__':
    import sys
    templatefile = sys.argv[1]
    modelbuilder = ModelBuilder(templatefile)

    print(modelbuilder.get_features())
    modelbuilder.showGraph()
    #a = Analyzer(templatefile)
    #a.getDeps()




from parser.Python3Listener import Python3Listener
from parser.Python3Parser import Python3Parser

class PyroFeatureParser(Python3Listener):
    def __init__(self):
        self.f=None
        self.insample = False
        self.post_map = {}
        self.curvar = ''
        pass

    def enterAtom(self, ctx):
        #print(ctx.getText())
        pass

    def enterFuncdef(self, ctx):
        self.f=str(ctx.NAME())

    def exitFuncdef(self, ctx):
        self.f=None

    def enterAtom_expr(self, ctx):

        if self.insample and ctx.atom().getText() == 'dist':

            if self.f is not None and self.f == 'guide':
                self.post_map[self.curvar]['post'] = ctx.trailer(0).NAME().getText()
            elif self.f is not None and self.f == 'model':

                self.post_map[self.curvar]['prior'] = ctx.trailer(0).NAME().getText()

    def enterExpr_stmt(self, ctx):
        if 'pyro.sample' in ctx.getText():
            self.insample = True
            self.curvar = ctx.testlist_star_expr(0).getText()
            if not self.curvar in self.post_map:
                self.post_map[self.curvar] = {}

    def exitExpr_stmt(self, ctx):
        if 'pyro.sample' in ctx.getText():
            self.insample = False

    def get_pyro_features(self, pyrofile_path):
        import antlr4
        from antlr4 import *
        from parser.Python3Parser import Python3Parser
        from parser.Python3Lexer import Python3Lexer

        pyrofile = antlr4.FileStream(pyrofile_path)
        lexer = Python3Lexer(pyrofile)
        stream = antlr4.CommonTokenStream(lexer)
        parser = Python3Parser(stream)

        code = parser.file_input()
        walker = ParseTreeWalker()
        walker.walk(self, code)
        feature_vector = {}
        for k in self.post_map:
            if 'post' in self.post_map[k]:
                feature_vector['vi_'+self.post_map[k]['prior'] + '_' + self.post_map[k]['post'] ] = 1
        return feature_vector


if __name__ == "__main__":
    import antlr4
    import sys
    from antlr4 import *
    from parser.Python3Parser import Python3Parser
    from parser.Python3Lexer import Python3Lexer

    from scipy.stats import *
    from numpy import *

    program=sys.argv[1]
    # results_file=sys.argv[2]
    # data_file=sys.argv[3]
    # try:
    #     output_file=sys.argv[4]
    # except:
    #     output_file="anyhow_this_file_doesnt_exist"
    template = antlr4.FileStream(program)
    lexer = Python3Lexer(template)
    stream = antlr4.CommonTokenStream(lexer)
    parser = Python3Parser(stream)
    featureParser = PyroFeatureParser()

    code = parser.file_input()
    walker = ParseTreeWalker()
    walker.walk(featureParser, code)
    print(featureParser.post_map)


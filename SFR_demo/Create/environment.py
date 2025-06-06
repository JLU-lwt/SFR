import generator,simplifier


class FunctionEnvironment(object):

    def __init__(self, params):
        self.params = params
        self.rng = None
        self.generator = generator.RandomFunctions(params)
        self.simplifier = simplifier.Simplifier(self.generator)
        

        self.equation_words=self.generator.equation_words

        self.id2word={i:s for i,s in enumerate(self.equation_words,4)}
        self.word2id={s:i for i,s in self.id2word.items()}
        self.word2id["P"] = 0
        self.word2id["S"] = 1
        self.word2id["F"] = 2
        self.word2id["CONSTANT"] = 3
        self.id2word[1] = "S"
        self.id2word[2] = "F"
        self.id2word[3] = "CONSTANT"
    def gen_expr(self):
        try:
            trees,dimension,info=self._gen_expr()
            if info:
                assert False
            return trees["tree"][0],trees["tree"][1],dimension
        except AssertionError:
            return None,None,None
        


    def _gen_expr(self):
        (F_tree,G_tree,dimension) = self.generator.generate_multi_dimensional_tree(self.rng)
        if F_tree is None or G_tree is None:
            return {"tree": [F_tree,G_tree]},dimension, ["bad tree"]

        if "x_j" not in G_tree.infix():
            return {"tree": [F_tree,G_tree]},dimension, ["no x_j"]

        for op in self.params.operators_to_not_repeat.split(","):
            if op and (F_tree.prefix().count(op) > 1 or G_tree.prefix().count(op) > 1):
                return {"tree": [F_tree,G_tree]}, dimension,["ops repeated"]

        if self.params.use_sympy:
            F_len_before = len(F_tree.prefix().split(","))
            G_len_before = len(G_tree.prefix().split(","))
            try:
                F_tree = self.simplifier.simplify_tree(F_tree)
                G_tree = self.simplifier.simplify_tree(G_tree)
            except:
                return {"tree": [F_tree,G_tree]}, dimension,["simplification error"]
            F_len_after = len(F_tree.prefix().split(","))
            G_len_after = len(G_tree.prefix().split(","))
            if F_tree is None or G_tree is None or F_len_after>2*F_len_before or G_len_after>2*G_len_before:
                return {"tree": [F_tree,G_tree]}, dimension,["simplification error"]
        

        trees, datapoints = self.generator.generate_datapoints_for_test(F_tree,G_tree,dimension)

        if datapoints is None:
            return {"tree": trees}, dimension, ["datapoint generation error"]
        
        return {"tree": trees}, dimension, []
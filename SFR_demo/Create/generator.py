import copy
import numpy as np
import numexpr as ne

from abc import ABC
from collections import defaultdict

from topo_utils import Topo




operators_real = {
    "add"   : 2,
    "sub"   : 2,
    "mul"   : 2,
    "div"   : 2,
    "abs"   : 1,
    "inv"   : 1,
    "sqrt"  : 1,
    "log"   : 1,
    "exp"   : 1,
    "sin"   : 1,
    "arcsin": 1,
    "cos"   : 1,
    "arccos": 1,
    "tan"   : 1,
    "arctan": 1,
    "pow2"  : 1,
    "pow3"  : 1,
    'id'    : 1
}

operators_extra = {"pow": 2}

math_constants = ["e", "pi"]
all_operators = {**operators_real, **operators_extra}

class Node:
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params
    
    def push_child(self, child):
        self.children.append(child)

    def prefix(self, skeleton=False):
        s = str(self.value)
        if skeleton:
            try: 
                float(s)
                s = "CONSTANT"
            except:
                pass
        for c in self.children:
            s += "," + c.prefix(skeleton=skeleton)
        return s
    
    def infix(self, skeleton=False):
        s = str(self.value)
        if skeleton:
            try: 
                if s.lstrip("-").isdigit():
                    s=s
                else:
                    float(s)
                    s = "CONSTANT"
            except:
                pass
        nb_children = len(self.children)
        if nb_children == 0:
            return s
        if nb_children == 1:
            if s == "pow2":
                s = "(" + self.children[0].infix(skeleton=skeleton) + ")**2"
            elif s == "inv":
                s = "1/(" + self.children[0].infix(skeleton=skeleton) + ")"
            elif s == "pow3":
                s = "(" + self.children[0].infix(skeleton=skeleton) + ")**3"
            else:
                s = s + "(" + self.children[0].infix(skeleton=skeleton) + ")"
            return s
        else:
            if s == "add":
                return self.children[0].infix(skeleton=skeleton) + " + " + self.children[1].infix(skeleton=skeleton)
            if s == "sub":
                return self.children[0].infix(skeleton=skeleton) + " - " + self.children[1].infix(skeleton=skeleton)
            if s == "pow":
                res  = "(" + self.children[0].infix(skeleton=skeleton) + ")**"
                res += ("" + self.children[1].infix(skeleton=skeleton))
                return res
            elif s == "mul":
                res  = "(" + self.children[0].infix(skeleton=skeleton) + ")" if self.children[0].value in ["add","sub"] else (self.children[0].infix(skeleton=skeleton))
                res += " * "
                res += "(" + self.children[1].infix(skeleton=skeleton) + ")" if self.children[1].value in ["add","sub"] else (self.children[1].infix(skeleton=skeleton))
                return res
            elif s == "div":
                res  = "(" + self.children[0].infix(skeleton=skeleton) + ")" if self.children[0].value in ["add","sub"] else (self.children[0].infix(skeleton=skeleton))
                res += " / "
                res += "(" + self.children[1].infix(skeleton=skeleton) + ")" if self.children[1].value in ["add","sub"] else (self.children[1].infix(skeleton=skeleton))
                return res







class Generator(ABC):
    def __init__(self, params):
        pass

    # @abstractmethod
    # def generate_datapoints(self, rng):
    #     pass


class RandomFunctions(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

        #方程维数
        self.min_dimension=params.min_dimension
        self.max_dimension=params.max_dimension

        #每维操作符数
        self.min_binary_ops_per_dim=params.min_binary_ops_per_dim
        self.max_binary_ops_per_dim=params.max_binary_ops_per_dim

        self.min_unary_ops_per_dim=params.min_unary_ops_per_dim
        self.max_unary_ops_per_dim=params.max_unary_ops_per_dim

        self.unary = False
        self.distrib = self.generate_dist(2 * self.max_binary_ops_per_dim * self.max_dimension)

        #操作符种类和概率
        self.operators = copy.deepcopy(operators_real)
        self.operators_downsample_ratio = defaultdict(float)
        for operator in self.params.operators_to_use.split(","):
            operator, ratio = operator.split(":")
            ratio = float(ratio)
            self.operators_downsample_ratio[operator] = ratio

        self.unaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 1]

        self.binaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 2]


        unaries_probabilities = []
        for op in self.unaries:
            if op not in self.operators_downsample_ratio:
                unaries_probabilities.append(0.0)
            else:
                ratio = self.operators_downsample_ratio[op]
                unaries_probabilities.append(ratio)
        self.unaries_probabilities = np.array(unaries_probabilities)
        if self.unaries_probabilities.sum()==0:
            self.use_unaries = False
        else:
            self.unaries_probabilities /= self.unaries_probabilities.sum()
            self.use_unaries = True

        binaries_probabilities = []
        for op in self.binaries:
            if op not in self.operators_downsample_ratio:
                binaries_probabilities.append(0.0)
            else:
                ratio = self.operators_downsample_ratio[op]
                binaries_probabilities.append(ratio)
        self.binaries_probabilities = np.array(binaries_probabilities)
        self.binaries_probabilities /= self.binaries_probabilities.sum()

        self.prob_const = params.prob_const
        self.prob_rand = params.prob_rand
        self.prob_t = params.prob_t
        self.prob_unaries = params.prob_unaries
        
        self.max_int = params.max_int

        self.constants_count=0


        self.constants = [
            str(i) for i in range(-3, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants

        self.variables = [f"x_i_{i}" for i in range(self.max_dimension)] + [f"x_j_{i}" for i in range(self.max_dimension)]
        self.symbols=(
            list(self.operators)
            + self.constants
            + self.variables
            + ["pow", "0"]
        )

        self.equation_words=sorted(list(set(self.symbols)))



        
    
    def generate_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = []
        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(
            len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1)
        ), "issue in generate_dist"
        return D


    def sample_next_pos(self, rng, nb_empty, nb_ops):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity

    def generate_float(self, rng, exponent=None):
        sign = rng.choice([-1, 1])
        constant = rng.uniform(np.log10(1/self.params.max_prefactor), np.log10(self.params.max_prefactor))
        constant = sign*10**constant
        return str(constant)

    def generate_leaf(self, rng, dimension,type):
        if rng.rand() < self.prob_rand:
            return "rand"
        elif rng.rand() < self.prob_t:
            return "t"
        else:
            draw = rng.rand()
            if draw < self.prob_const:
                return self.generate_int(rng)
            else:
                dimension = rng.randint(0, dimension)
                if type=="F":
                    return f"x_i_{dimension}"
                elif type=="G":
                    return rng.choice([f"x_i_{dimension}", f"x_j_{dimension}"])



    def generate_ops(self, rng, arity):
        if arity == 1:
            ops = self.unaries
            probas = self.unaries_probabilities
        else:
            ops = self.binaries
            probas = self.binaries_probabilities
        return rng.choice(ops, p=probas)


    def generate_tree(self, rng, nb_binary_ops, dimension,type):
        tree = Node(0, self.params)
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1

        while nb_binary_ops > 0:
            next_pos, arity = self.sample_next_pos(rng, nb_empty, nb_binary_ops)
            next_en += next_pos
            op = self.generate_ops(rng, arity)
            
            empty_nodes[next_en].value = op
            for _ in range(arity):
                e = Node(0, self.params)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            next_en += 1
            nb_empty += arity - 1 - next_pos
            nb_binary_ops -= 1
        rng.shuffle(empty_nodes)
        for n in empty_nodes:
            if len(n.children) == 0:
                n.value = self.generate_leaf(rng, dimension,type)
        return tree
    

    def generate_multi_dimensional_tree(self,rng):
        dimension=rng.randint(self.min_dimension,self.max_dimension+1)

        nb_binary_ops_to_use = rng.randint(self.min_binary_ops_per_dim, self.max_binary_ops_per_dim+1)

        nb_unary_ops_to_use = rng.randint(self.min_unary_ops_per_dim, self.max_unary_ops_per_dim + 1)
    
        F_tree = self.generate_tree(rng, nb_binary_ops_to_use, dimension,"F")

        G_tree = self.generate_tree(rng, nb_binary_ops_to_use, dimension,"G")
        
        if self.use_unaries:
            F_tree = self.add_unaries(rng, F_tree, nb_unary_ops_to_use)
            G_tree = self.add_unaries(rng, G_tree, nb_unary_ops_to_use)
            
        if self.params.reduce_num_constants:
            F_tree = self.add_prefactors(rng, F_tree)
            G_tree = self.add_prefactors(rng, G_tree)
        
        return F_tree,G_tree,dimension


    def prefix_to_tree(self,prefix):
        tree=self._prefix_to_tree(prefix)[0]
        return tree
    
    def _prefix_to_tree(self,lst):


        if len(lst) == 0:
            return None, 0
        
        elif lst[0] in all_operators.keys():
            res = Node(lst[0], self.params)
            arity = all_operators[lst[0]]
            pos = 1
            for i in range(arity):
                child, length = self._prefix_to_tree(lst[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        elif lst[0] in self.symbols:
            return Node(lst[0], self.params), 1
        else:
            try:
                float(lst[0])  # if number, return leaf
                return Node(lst[0], self.params), 1
            except:
                return None, 0


    def add_unaries(self, rng, tree, nb_unaries):
        prefix = self._add_unaries(rng, tree)
        prefix = prefix.split(",")

        indices = []
        for i, x in enumerate(prefix):
            if x in self.unaries:
                indices.append(i)
        rng.shuffle(indices)
        if len(indices) > nb_unaries:
            to_remove = indices[: len(indices) - nb_unaries]
            for index in sorted(to_remove, reverse=True):
                del prefix[index]

        tree=self.prefix_to_tree(prefix)

        return tree

    def _add_unaries(self, rng, tree):

        s = str(tree.value)
        for c in tree.children:
            if len(c.prefix().split(",")) < self.params.max_unary_depth and rng.rand() < self.params.prob_unaries:
                unary = rng.choice(self.unaries, p=self.unaries_probabilities)
                if unary=='id': s += "," + self._add_unaries(rng, c)
                s += f",{unary}," + self._add_unaries(rng, c)
            else:
                s += f"," + self._add_unaries(rng, c)
        return s


    def add_prefactors(self, rng, tree):
        transformed_prefix = self._add_prefactors(rng, tree)
        tree=self.prefix_to_tree(transformed_prefix.split(","))
        return tree

    def _add_prefactors(self, rng, tree):
        
        
        s = str(tree.value)
        a, b, c = [self.generate_float(rng) for _ in range(3)]
        # add_prefactor = f",add,cA," if rng.rand() < self.params.prob_prefactor else ","
        # mul_prefactor1 = f",mul,cM1," if rng.rand() < self.params.prob_prefactor else ","
        # mul_prefactor2 = f",mul,cM2," if rng.rand() < self.params.prob_prefactor else ","
        add_prefactor = f",add,{a}," if rng.rand() < self.params.prob_prefactor else ","
        mul_prefactor1 = f",mul,{b}," if rng.rand() < self.params.prob_prefactor else ","
        mul_prefactor2 = f",mul,{c}," if rng.rand() < self.params.prob_prefactor else ","
        total_prefactor = add_prefactor.rstrip(",") + "," + mul_prefactor1.lstrip(",")
        if s in ["add", "sub"]:
            s += (
                "," if tree.children[0].value in ["add", "sub"] else mul_prefactor1
            ) + self._add_prefactors(rng, tree.children[0])
            s += (
                "," if tree.children[1].value in ["add", "sub"] else mul_prefactor2
            ) + self._add_prefactors(rng, tree.children[1])
        elif s in self.unaries and tree.children[0].value not in ["add", "sub"]:
            s += total_prefactor + self._add_prefactors(rng, tree.children[0])
        else:
            for c in tree.children:
                s += f"," + self._add_prefactors(rng, c)
        return s


    def generate_datapoints_for_test(self,F_tree,G_tree,dimension):
        topo_flag = False
        while not topo_flag:
            topo_nodes = np.random.randint(int(self.params.topo_max_nodes / 10), self.params.topo_max_nodes)
            topo_type = np.random.choice(a=self.params.topo_type_list, size=1, replace=None)
            topo = Topo(N=topo_nodes, topo_type=topo_type)
            topo_nodes = topo.N
            if all(node in topo.sparse_adj[0] for node in range(0, topo.N)):
                topo_flag = True
        
        n_points=self.params.n_points
        node_state=np.random.normal(loc=0,scale=1,size=(n_points,topo.N,dimension))
        # node_state=np.array(torch.rand(n_points,topo.N,dimension),dtype='float64')
        F_vals,G_vals=integrate_value(node_state,F_tree,G_tree,topo,dimension)
        if F_vals is None or G_vals is None:
            return [F_tree,G_tree],None
        
        if np.any(np.isnan(F_vals)) or np.any(np.isnan(G_vals)):
            return [F_tree,G_tree], None
        if np.any(np.abs(F_vals)>10**self.params.max_exponent) or np.any(np.abs(G_vals)>10**self.params.max_exponent):
            return [F_tree,G_tree], None

        return [F_tree,G_tree],[F_vals,G_vals]


def _integrate_value(node_state,F_tree,G_tree,topo,dimension):
    tree=tree_to_numexpr_fn(F_tree,G_tree,topo,dimension)
    F_vals,G_vals=tree(node_state)
    return F_vals,G_vals


def integrate_value(node_state,F_tree,G_tree,topo,dimension):
        try:
            return _integrate_value(node_state,F_tree,G_tree,topo,dimension)
        except:
            return None,None
        
        
    
def tree_to_numexpr_fn(F_tree,G_tree,topo,dim):
        if not isinstance(F_tree, str) and not isinstance(G_tree, str):
            F_infix = F_tree.infix()
            G_infix = G_tree.infix()
        else:
            F_infix = F_tree
            G_infix = G_tree
        
        row,col=topo.sparse_adj
        def wrapped_numexpr_fn(x):
            F_local_dict={}
            G_local_dict={}
            try:
                for d in range(dim):
                    F_local_dict["x_i_{}".format(d)] = x[:,:, d].reshape(-1,1)
                    G_local_dict["x_i_{}".format(d)] = x[:,col, d].reshape(-1,1)
                    G_local_dict["x_j_{}".format(d)] = x[:,row, d].reshape(-1,1)

                F_vals = ne.evaluate(F_infix, local_dict=F_local_dict)
                G_vals = ne.evaluate(G_infix, local_dict=G_local_dict)
            except Exception as e:
                F_vals=None
                G_vals=None
            return F_vals,G_vals
        return wrapped_numexpr_fn

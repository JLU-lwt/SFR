import copy

from abc import ABC, abstractmethod



class FunctionEnvironment(object):

    def __init__(self, params):
        self.params = params
        self.rng = None
        self.generator = RandomFunctions(params)




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
                if s.lstrip("-").isdigit():
                    s=s
                else:
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
        self.max_dimension=params.max_dimension
        self.operators = copy.deepcopy(operators_real)
        self.max_int = params.max_int
        self.constants = [
            str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants

        self.variables = ["rand"] + [f"x_i_{i}" for i in range(self.max_dimension)] + [f"x_j_{i}" for i in range(self.max_dimension)] + ["t"] +["CONSTANT"]

        self.symbols=(
            list(self.operators)
            + self.constants
            + self.variables
            + ["INT+", "INT-","pow", "0"]
        )

    

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
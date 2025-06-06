from abc import ABC
import sympy as sp
from sympy.core.rules import Transform
from contextlib import contextmanager
import signal
from generator import all_operators
class InvalidPrefixExpression(BaseException):
    pass

@contextmanager
def timeout(time):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        pass
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError





class Simplifier(ABC):

    local_dict = {
        "n": sp.Symbol("n", real=True, nonzero=True, positive=True, integer=True),
        "e": sp.E,
        "pi": sp.pi,
        "euler_gamma": sp.EulerGamma,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "step": sp.Heaviside,
        "sign": sp.sign,
    }
    for d in range(10):
        k = "x_{}".format(d)
        local_dict[k] = sp.Symbol(k, real=True, integer=False)

    def __init__(self, generator):
        self.generator=generator
        self.params = generator.params

        for k in generator.variables:
            self.local_dict[k] = sp.Symbol(k, real=True, integer=False)
    
    def simplify_tree(self, tree):
        
        if tree is None:
            return tree
        expr = self.tree_to_sympy_expr(tree)
        new_tree = self.sympy_expr_to_tree(expr)
        if new_tree is None:
            return tree
        else:
            return new_tree
    

    @classmethod
    def tree_to_sympy_expr(cls, tree, round=True):
        prefix = tree.prefix().split(",")
        
        sympy_compatible_infix = cls.prefix_to_sympy_compatible_infix(prefix)
        expr = sp.parse_expr(
            sympy_compatible_infix, evaluate=True, local_dict=cls.local_dict
        )
        
        if round: 
            expr = cls.round_expr(expr)
        return expr
    

    @classmethod
    def _prefix_to_sympy_compatible_infix(cls, expr):
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in all_operators:
            args = []
            l1 = expr[1:]
            for _ in range(all_operators[t]):
                i1, l1 = cls._prefix_to_sympy_compatible_infix(l1)
                args.append(i1)
            return cls.write_infix(t, args), l1
        else:  # leaf
            try:
                float(t)
                t = str(t)
            except ValueError:
                t = t
            return t, expr[1:]


    @classmethod
    def prefix_to_sympy_compatible_infix(cls, expr):
        p, r = cls._prefix_to_sympy_compatible_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"
    
    @classmethod
    def round_expr(cls, expr, decimals=4):

        expr = expr.xreplace(
            Transform(
                lambda x: x.round(decimals), lambda x: isinstance(x, sp.Float)
            )
        )
        return expr
    
    def sympy_expr_to_tree(self, expr):
        
        prefix = self.sympy_to_prefix(expr)
        return self.generator.prefix_to_tree(prefix)



    @classmethod
    def write_infix(cls, token, args):
        """
        Infix representation.
    
        """
        if token == "add":
            return f"({args[0]})+({args[1]})"
        elif token == "sub":
            return f"({args[0]})-({args[1]})"
        elif token == "mul":
            return f"({args[0]})*({args[1]})"
        elif token == "div":
            return f"({args[0]})/({args[1]})"
        if token == "pow":
            return f"({args[0]})**({args[1]})"
        elif token == "idiv":
            return f"idiv({args[0]},{args[1]})"
        elif token == "mod":
            return f"({args[0]})%({args[1]})"
        elif token == "abs":
            return f"Abs({args[0]})"
        elif token == "id":
            return f"{args[0]}"
        elif token == "inv":
            return f"1/({args[0]})"
        elif token == "pow2":
            return f"({args[0]})**2"
        elif token == "pow3":
            return f"({args[0]})**3"
        elif token in all_operators:
            return f"{token}({args[0]})"
        else:
            return token
        raise InvalidPrefixExpression(
            f"Unknown token in prefix expression: {token}, with arguments {args}"
        )
    



    def _sympy_to_prefix(self, op, expr):
        n_args = len(expr.args)
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]
        elif isinstance(expr, sp.Float):
            s = str(expr)
            return [s]
        elif isinstance(expr, sp.Rational):
            return ["mul", str(expr.p), "pow", str(expr.q), "-1"]
        elif expr == sp.EulerGamma:
            return ["euler_gamma"]
        elif expr == sp.E:
            return ["e"]
        elif expr == sp.pi:
            return ["pi"]

        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)

        # Unknown operator
        return self._sympy_to_prefix(str(type(expr)), expr)

    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Mod: "mod",
        sp.Pow: "pow",
        # Misc
        sp.Abs: "abs",
        sp.sign: "sign",
        sp.Heaviside: "step",
        # Exp functions
        sp.exp: "exp",
        sp.log: "log",
        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        # Trigonometric Inverses
        sp.asin: "arcsin",
        sp.acos: "arccos",
        sp.atan: "arctan",
    }
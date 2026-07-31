"""
parser.py
---------

Stage 2 of the Symbolic Transfer Learning workflow.

Responsibilities
----------------
1. Convert PySR expressions into executable callables.
2. Support arbitrary nesting.
3. Automatically determine variables used.
4. Store an expression tree for future analysis.
5. Easy to extend with additional operators.

Supported grammar
-----------------
Binary:
    +  -  *  /  ^

Unary:
    sin
    cos
    exp

Variables:
    x0, x1, ...

Constants:
    Any integer or float.

"""

import ast
import operator
import numpy as np


# ============================================================
# Operator registry
# ============================================================

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
}


# ============================================================
# Expression Tree Nodes
# ============================================================

class Node:
    pass


class ConstantNode(Node):

    def __init__(self, value):
        self.value = float(value)

    def __repr__(self):
        return f"{self.value}"


class VariableNode(Node):

    def __init__(self, index):
        self.index = int(index)

    def __repr__(self):
        return f"x{self.index}"


class UnaryNode(Node):

    def __init__(self, name, child):
        self.name = name
        self.child = child

    def __repr__(self):
        return f"{self.name}({self.child})"


class BinaryNode(Node):

    def __init__(self, op, left, right):

        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):

        return f"({self.left} {self.op} {self.right})"


# ============================================================
# AST -> ExpressionTree Converter
# ============================================================

class ASTConverter:

    def __init__(self):

        self.variables = set()

    # --------------------------------------------------------

    def convert(self, expression):

        """
        Parameters
        ----------
        expression : str

            Original PySR expression.

        Returns
        -------
        Node
        """

        expression = expression.replace("^", "**")

        tree = ast.parse(expression, mode="eval")

        return self.visit(tree.body)

    # --------------------------------------------------------

    def visit(self, node):

        if isinstance(node, ast.BinOp):

            return self.visit_binary(node)

        elif isinstance(node, ast.Call):

            return self.visit_call(node)

        elif isinstance(node, ast.Name):

            return self.visit_name(node)

        elif isinstance(node, ast.Constant):

            return ConstantNode(node.value)

        elif isinstance(node, ast.Num):

            return ConstantNode(node.n)

        elif isinstance(node, ast.UnaryOp):

            return self.visit_unary(node)

        else:

            raise TypeError(
                f"Unsupported AST node: {type(node)}"
            )

    # --------------------------------------------------------

    def visit_binary(self, node):

        op_type = type(node.op)

        if op_type not in BINARY_OPERATORS:

            raise ValueError(
                f"Unsupported binary operator: {op_type}"
            )

        symbol = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "^"
        }[op_type]

        left = self.visit(node.left)
        right = self.visit(node.right)

        return BinaryNode(
            symbol,
            left,
            right
        )

    # --------------------------------------------------------

    def visit_call(self, node):

        if not isinstance(node.func, ast.Name):

            raise ValueError(
                "Unsupported function."
            )

        name = node.func.id

        if name not in UNARY_OPERATORS:

            raise ValueError(
                f"Unsupported unary operator: {name}"
            )

        if len(node.args) != 1:

            raise ValueError(
                f"{name} expects exactly one argument."
            )

        child = self.visit(node.args[0])

        return UnaryNode(
            name,
            child
        )

    # --------------------------------------------------------

    def visit_name(self, node):

        name = node.id

        if not name.startswith("x"):

            raise ValueError(
                f"Illegal variable name: {name}"
            )

        index = int(name[1:])

        self.variables.add(index)

        return VariableNode(index)

    # --------------------------------------------------------

    def visit_unary(self, node):

        if isinstance(node.op, ast.USub):

            child = self.visit(node.operand)

            return BinaryNode(
                "*",
                ConstantNode(-1.0),
                child
            )

        raise ValueError(
            "Unsupported unary operator."
        )
      # ============================================================
# Tree Compiler
# ============================================================

class TreeCompiler:

    def __init__(self):
        pass

    # --------------------------------------------------------

    def compile(self, root):

        """
        Compile an expression tree into a callable.

        Returns
        -------
        function(X)

        where

        X.shape = (n_samples, n_features)
        """

        def func(X):
            return self.evaluate(root, X)

        return func

    # --------------------------------------------------------

    def evaluate(self, node, X):

        if isinstance(node, ConstantNode):
            #return node.value - Should be fixed!!
            # Broadcast scalar constant over all samples
            return np.full(X.shape[0], node.value, dtype=float)

        elif isinstance(node, VariableNode):
            return X[:, node.index]

        elif isinstance(node, UnaryNode):

            child = self.evaluate(node.child, X)

            return UNARY_OPERATORS[node.name](child)

        elif isinstance(node, BinaryNode):

            left = self.evaluate(node.left, X)
            right = self.evaluate(node.right, X)

            if node.op == "+":
                return left + right

            elif node.op == "-":
                return left - right

            elif node.op == "*":
                return left * right

            elif node.op == "/":
                return left / right

            elif node.op == "^":
                return left ** right

            else:

                raise ValueError(
                    f"Unknown operator {node.op}"
                )

        else:

            raise TypeError(
                f"Unknown node type {type(node)}"
            )


# ============================================================
# Utilities
# ============================================================

def extract_variables(root):

    """
    Return sorted variable indices used in the tree.
    """

    variables = set()

    def walk(node):

        if isinstance(node, VariableNode):

            variables.add(node.index)

        elif isinstance(node, UnaryNode):

            walk(node.child)

        elif isinstance(node, BinaryNode):

            walk(node.left)
            walk(node.right)

    walk(root)

    return sorted(variables)


# ------------------------------------------------------------

def validate_dataset(root, X):

    """
    Check whether the supplied dataset contains
    enough columns.
    """

    variables = extract_variables(root)

    if len(variables) == 0:
        return

    required = max(variables) + 1

    if X.shape[1] < required:

        raise ValueError(

            f"Expression requires {required} variables "
            f"(x0 ... x{required-1}) but dataset "
            f"contains only {X.shape[1]} columns."

        )


# ============================================================
# Public Parser
# ============================================================

class ExpressionParser:

    """
    Main parser interface.
    """

    def __init__(self):

        self.converter = ASTConverter()

        self.compiler = TreeCompiler()

    # --------------------------------------------------------

    def parse(self, expression):

        """
        Convert PySR expression into expression tree.
        """

        self.converter.variables.clear()

        tree = self.converter.convert(expression)

        variables = sorted(self.converter.variables)

        return tree, variables

    # --------------------------------------------------------

    def compile(self, expression):

        """
        Convert expression directly into callable.
        """

        tree, variables = self.parse(expression)

        function = self.compiler.compile(tree)

        return function, tree, variables

    # --------------------------------------------------------

    def compile_candidate(self, candidate):

        """
        Compile a CandidateExpression object.

        Required fields

            candidate.equation

        Generated fields

            candidate.function
            candidate.tree
            candidate.variables

        """

        function, tree, variables = self.compile(
            candidate.equation
        )

        candidate.function = function

        candidate.tree = tree

        candidate.variables = variables

        return candidate

    # --------------------------------------------------------

    def evaluate(self, expression, X):

        """
        Direct evaluation without explicitly creating
        a CandidateExpression object.
        """

        function, tree, variables = self.compile(
            expression
        )

        validate_dataset(tree, X)

        return function(X)


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    expr = "sin(x0)+x1^2-exp(x2/x3)"

    parser = ExpressionParser()

    function, tree, variables = parser.compile(expr)

    print("Tree")
    print(tree)

    print()

    print("Variables")
    print(variables)

    X = np.random.rand(10, 4)

    validate_dataset(tree, X)

    y = function(X)

    print()

    print("Prediction")

    print(y)

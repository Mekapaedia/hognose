#!/usr/bin/env python3

import sys
from lark import Lark, Transformer

class NameRef:
    def __init__(self, name):
        self.name = name

class AstBuilder(Transformer):
    symbol_table = {}
    def start(self, *args):
        print(self.symbol_table)

    def arith(self, *args):
        if len(args) == 1:
            arg = args[0][0]
            if isinstance(arg, NameRef):
                return self.symbol_table[arg.name]
            else:
                return arg
        else:
            print(*args)
            return args

    def assign(self, *args):
        target, assign_op, assign_expr = args[0]
        target = target.name
        if assign_op == "=":
            self.symbol_table[target] = assign_expr
            return self.symbol_table[target]
        else:
            raise Exception("idk is " + assign_op)
        print(*args)
        return args

    def sum(self, *args):
        lhs, sum_op, rhs = args[0]
        if isinstance(lhs, NameRef):
            lhs = self.symbol_table[lhs.name]
        if isinstance(rhs, NameRef):
            rhs = self.symbol_table[rhs.name]
        sum_op = sum_op.value
        if sum_op == "+":
            return lhs + rhs

        raise Exception("idk is " + sum_op)

    def product(self, *args):
        lhs, prod_op, rhs = args[0]
        if isinstance(lhs, NameRef):
            lhs = self.symbol_table[lhs.name]
        if isinstance(rhs, NameRef):
            rhs = self.symbol_table[rhs.name]
        prod_op = prod_op.value
        if prod_op == "/":
            return lhs / rhs

        raise Exception("idk is " + prod_op)

    def name(self, *args):
        return NameRef(args[0][0].value)

    def integer(self, *args):
        return int(args[0][0])

    def assign_op(self, *args):
        return args[0][0].value

    def __default__(self, *args):
        print(*args)
        return args

grammar_text = None

with open("grammar.lark") as f:
    grammar_text = f.read()

parser = Lark(grammar_text)

input_text = None

with open(sys.argv[1]) as f:
    input_text = f.read()

parse_res = parser.parse(input_text)
print(parse_res.pretty())
AstBuilder().transform(parse_res)

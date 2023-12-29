#!/usr/bin/env python

from lark import Lark, Tree
from lark.visitors import Interpreter

grammar = Lark(
r'''
program: stmt+
stmt: expression endstmt
expression: assignment | block | sum
block: OPENBRACE stmt+ CLOSEBRACE
assignment: [storagetype] primary [typebound] rhs_assign
typebound: COLON sum
rhs_assign: assign_opt expression
sum: sum sum_opt term
   | term
term: term term_opt primary
    | primary
primary: atom | parenexp
atom: binding | literal
parenexp: OPENPAREN expression CLOSEPAREN

binding: NAME
literal: INTEGER | FLOAT | DQSTRING | TDQSTRING | SQSTRING | TSQSTRING
storagetype: STATIC | CONST
sum_opt: PLUS | MINUS
term_opt: MULT | DIVIDE
assign_opt: EQUAL
endstmt: SEMICOLON | NEWLINE

STATIC: "static"
CONST: "const"
DYNAMIC: "dynamic"

PLUS: "+"
MINUS: "-"
MULT: "*"
DIVIDE: "/"
EQUAL: "="
BANG: "!"
DOT: "."
COMMA: ","
OPENBRACE: "{"
CLOSEBRACE: "}"
OPENSQUARE: "["
CLOSESQUARE: "]"
OPENPAREN: "("
CLOSEPAREN: ")"
COLON: ":"
SEMICOLON: ";"
NEWLINE: "\n"
SPACE: " "
TAB: "\t"

DQSTRING: /(r|b){0,2}\"([^\"\\]|\\.)*\"/
TDQSTRING: /(r|b){0,2}\"{3}([^\\]|\\.)*\"{3}/
SQSTRING: /(r|b){0,2}\'([^\'\\]|\\.)*\'/
TSQSTRING: /(r|b){0,2}\'{3}([^\\]|\\.)*\'{3}/
FLOAT: /[-+]?[0-9_]+\.[0-9_]+([eE][+-][0-9_]+)?/
INTEGER: /(0|[+-]?[1-9][0-9_]*)/
NAME: /[_a-z][_a-z0-9]*/i

%ignore SPACE
%ignore TAB
''', start="program", propagate_positions=True)

class Binding:
    def __init__(self, name, storage_type, type_bound, value_reference=None):
        self.name = name
        self.storage_type = storage_type
        self.type_bound = type_bound
        self.value_reference = None
        self.set_reference(value_reference)

    def set_reference(self, value_reference):
        if self.value_reference is not None:
            if self.storage_type == "const":
                raise Exception("Attempted to assign value '{value}' to const reference '{name}'".format(
                    value=value_reference, name=self.name))
            if self.storage_type == "static" and not self.typematch(value_reference.value_type):
                raise Exception("Attempted to assign value '{value}' of type '{newtype}' to static reference '{name}' of non-matching type '{curtype}'".format(
                    value=value_reference, newtype=value_reference.value_type, name=self.name, curtype=self.value_type))
        self.value_reference = value_reference
        self.value_type = self.value_reference.value_type if value_reference is not None else None
        print("Assignment: '{}' is now '{}'".format(self.name, self.value_reference))

    def assign(self, assign_op, value_reference):
        if assign_op == "=":
            self.set_reference(value_reference)
        else:
            raise Exception("Unknown assignment operator '{}'".format(assign_op))

    def get_reference(self):
        print("Reading reference '{}'".format(self))
        return self.value_reference

    def typematch(self, value_type):
        return self.value_type == value_type # add actual type bounds here

    def __repr__(self):
        return "{} {}: {} = {}".format(self.storage_type, self.name, self.type_bound, self.value_reference)

    def __str__(self):
        return self.__repr__()

class HognoseEval(Interpreter):

    def __init__(self):
        super().__init__()
        self.symbol_table = SymbolTable()

    def visit_or_default(self, tree, default):
        if tree is None:
            return default
        else:
            return self.visit(tree)

    def program(self, tree):
        return self.visit_children(tree)

    def stmt(self, tree):
        return self.visit(tree.children[0])

    def expression(self, tree):
        return self.visit(tree.children[0])

    def assignment(self, tree):
        pass

    def rhs_assign(self, tree):
        pass

    def assign_opt(self, tree):
        return tree.children[0].value

    def sum(self, tree):
        if len(tree.children) == 3:
            sum_lhs = self.get_symbol_or_value(self.visit(tree.children[0]))
            sum_opt = self.visit(tree.children[1])
            sum_rhs = self.get_symbol_or_value(self.visit(tree.children[2]))
            return sum_lhs.binop(sum_opt, sum_rhs)
        else:
            return self.visit(tree.children[0])

    def sum_opt(self, tree):
        return tree.children[0].value

    def term(self, tree):
        if len(tree.children) == 3:
            term_lhs = self.get_symbol_or_value(self.visit(tree.children[0]))
            term_opt = self.visit(tree.children[1])
            term_rhs = self.get_symbol_or_value(self.visit(tree.children[2]))
            return term_lhs.binop(term_opt, term_rhs)
        else:
            return self.visit(tree.children[0])

    def term_opt(self, tree):
        return tree.children[0].value

    def primary(self, tree):
        return self.visit(tree.children[0])

    def atom(self, tree):
        return self.visit(tree.children[0])

    def literal(self, tree):
        if tree.children[0].type == "INTEGER":
            return Value(int(tree.children[0].value))
        elif tree.children[0].type == "FLOAT":
            return Value(float(tree.children[0].value))

    def binding(self, tree):
        return str(tree.children[0].value)

    def storagetype(self, tree):
        return str(tree.children[0].value)

parse_tree = grammar.parse("static a = 2 + 3\nconst c = 3 - 2/a;")
HognoseEval().visit(parse_tree)

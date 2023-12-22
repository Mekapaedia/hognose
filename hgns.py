#!/usr/bin/env python

from lark import Lark

grammar = Lark(
r'''
program: stmt+
stmt: expr WS? endstmt
expr: decl | assign | arith
decl: vartype WS? NAME WS? lhs_assign?
assign: NAME WS? lhs_assign
lhs_assign: assign_opt WS? expr
arith: sum
sum: innersum | term
innersum: sum WS? sum_opt WS? term
term: innerterm | primary
innerterm: term WS? mul_opt WS? primary
primary: parenexpr | NAME | literal
parenexpr: OPENPAREN WS? expr WS? CLOSEPAREN
literal: NUMBER

vartype: VAR | CONST
sum_opt: PLUS | MINUS
mul_opt: MULT | DIVIDE
assign_opt: EQUAL
endstmt: SEMICOLON | NEWLINE

VAR: "var"
CONST: "const"
PLUS: "+"
MINUS: "-"
MULT: "*"
DIVIDE: "\/"
EQUAL: "="
OPENPAREN: "("
CLOSEPAREN: ")"
SEMICOLON: ";"
NEWLINE: "\n"
NUMBER: /(0|[1-9][0-9]*)/
NAME: /[_a-z][_a-z0-9]*/i
WS: /[ \t\n]+/

''', start="program")

print(grammar)
print(grammar.parse("var a = 2;"))

#!/usr/bin/env python3

from lark import Lark

grammar_text = None

with open("grammar.lark") as f:
    grammar_text = f.read()

parser = Lark(grammar_text)

input_text = """
[2: 500_000.2e-10+3i-5.0j+400k, 3[1..2]: [2...4;[2..a[1..3]]]]

2
3
"""

print(parser.parse(input_text).pretty())

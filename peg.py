#!/usr/bin/env python

import sys
from parsimonious.grammar import Grammar

grammar_text = None

with open("grammar.ppeg") as f:
    grammar_text = f.read()

parser = Grammar(grammar_text)

input_text = None

with open(sys.argv[1]) as f:
    input_text = f.read()

parse_tree = parser.parse(input_text)
print(parse_tree)

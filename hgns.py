#!/usr/bin/env python

from arpeggio.cleanpeg import ParserPEG
import sys

grammar_file_lines = ""
with open("grammar.g") as grammar_file:
    grammar_file_lines = grammar_file.read()

hgns_parser = ParserPEG(grammar_file_lines, "Program", debug=True)

prog_file_lines = ""
with open(sys.argv[1]) as prog_file:
    prog_file_lines = prog_file.read()

parse_tree = hgns_parser.parse(prog_file_lines)
print(parse_tree)

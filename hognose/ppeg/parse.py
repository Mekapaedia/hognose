import os
from ..parser import BaseParser, ParserError, InternalParserError
from parsimonious.grammar import Grammar

DEFAULT_GRAMMAR_NAME = "grammar.ppeg"
DEFAULT_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, DEFAULT_GRAMMAR_NAME)

class PpegParser(BaseParser):
    def __init__(self, grammar_file=DEFAULT_GRAMMAR_PATH):
        super().__init__(grammar_file)
        self.parser = Grammar(self.grammar_text)

    def parse(self, text, print_tree=False):
        parse_res = self.parser.parse(text)
        if print_tree is True:
            print(parse_res)
        raise NotImplementedError

#!/usr/bin/env python3

import sys
import re
from lark import Lark, Transformer, Tree
from lark.lexer import Lexer, Token
from lark.exceptions import UnexpectedCharacters, LexError, ParseError
from lark.visitors import Transformer_InPlace
rich_imported = False
try:
    import rich
    rich_imported = True
except ImportError:
    pass

def tree_print(tree):
    if rich_imported:
        rich.print(tree)
    else:
        print(tree.pretty())

class LexMatch:
    def __init__(self, terminal):
        self.pattern = re.compile(terminal.pattern.to_regexp())
        self.priority = terminal.priority

class HognoseLexer(Lexer):
    def __init__(self, lexer_conf):
        self.ignore = lexer_conf.ignore
        self.patterns = {}
        for terminal in lexer_conf.terminals:
            self.patterns[terminal.name] = LexMatch(terminal)

    def get_token_match(self, data):
        matches = {}
        highest_prio = None
        for term_name, lex_match in self.patterns.items():
            match = lex_match.pattern.match(data)
            if match:
                match = match.group(0)
                prio = lex_match.priority
                match_len = len(match)
                if highest_prio is None or prio > highest_prio:
                    highest_prio = prio
                if prio not in matches:
                    matches[prio] = {"longest": None}
                match_dict = matches[prio]
                if match_dict["longest"] is None or match_len > match_dict["longest"]:
                    match_dict["longest"] = match_len
                if match_len not in match_dict:
                    match_dict[match_len] = []
                new_token = Token(type=term_name,
                                  value=match,
                                  start_pos=self.pos,
                                  end_pos=self.pos+match_len,
                                  line=self.line,
                                  end_line=self.line+str(match).count('\n'),
                                  column=self.col,
                                  end_column=(self.col if str(match).count('\n') == 0 else 1)+match_len
                                  )

                match_dict[match_len].append(new_token)
        if highest_prio is not None:
            lex_match = matches[highest_prio][matches[highest_prio]["longest"]]
            if len(lex_match) > 1:
                raise LexError("Multiple equal length matches: {} in the current parser context, at line {} col {}".format(
                               [(x.type, x.value) for x in lex_match], self.line, self.col))
            else:
                return lex_match[0]
        else:
            raise UnexpectedCharacters(data, self.pos, self.line, self.column)

    def lex(self, data):
        tokens = []
        self.col = 1
        self.line = 1
        self.pos = 0
        while len(data) > 0:
            match = self.get_token_match(data)
            data = data[len(str(match.value)):]
            self.col = match.end_column
            self.line = match.end_line
            self.pos = match.end_pos
            if match.type not in self.ignore:
                tokens.append(match)
        while len(tokens):
            yield tokens.pop(0)

class HognosePostParse(Transformer_InPlace):

    def _ambig_cond_body(self, tree):
        tighter_bound_group = [x for x in tree if x.children[2] is None]
        less_bound_group = [x for x in tree if x.children[2] is not None]
        if len(tighter_bound_group) == 1:
            return tighter_bound_group[0]
        raise ParseError("Don't know how to deal with this ambiguity yet: {}".format(tree))

    def _ambig(self, tree):
        if len(tree) == 1:
            return tree[0]
        ambig_names = set([x.data.value for x in tree])
        if len(ambig_names) > 1:
            raise ParseError("Cannot resolve ambiguity for more than one rule type: {}".format(list(ambig_names)))
        ambig_name = list(ambig_names)[0]
        ambig_handler = "_ambig_{}".format(ambig_name)
        if not hasattr(self, ambig_handler):
            raise ParseError("No ambiguity handler registered for {}\n{}".format(ambig_name, tree))
        return getattr(self, ambig_handler)(tree)

    def term_expr(self, tree):
        pass

    def newlines(self, tree):
        pass

grammar_text = None

with open("grammar.lark") as f:
    grammar_text = f.read()

parser = Lark(grammar_text, propagate_positions=True, ambiguity="explicit", lexer=HognoseLexer)

input_text = None

input_texts = [
    """
    cheese,fries;cheese;{cheese;cheese,fries;}
    """,
    """
    cheese,
    fries;{
    cheese
    cheese, fries
    }
    cheese
    """,
    """
    cheese,
    fries
    cheese
    {cheese
    cheese,fries;}
    """,
    """
    if fries
    chicken
    """,
    """
    if fries
    {
        cheese
        cheese, chicken
    }
    elif cheese
        if chicken fries else cheese
    """,
    """
    a = {
        if h
            h
        else
            c
    }
    """,
    """
    a = for cheese, fries in chicken
        cheeses
    elif chicken_fries
        waaaa
    elwhile try
        noooo
    else
        sa
    """,
    """
    if a
        if b
            b
        else
            if d
                d
            else
                e
    """,
    """
    if a
        if b
            b
        elif c
            c
        else
            if d
                d
            else
                e
    elfor b in c
        b
    """

]

#with open(sys.argv[1]) as f:
#    input_text = f.read()

if input_text is None:
    for input_text in input_texts:
        print("-"*80)
        print(input_text)
        parse_res = HognosePostParse().transform(parser.parse(input_text))
        tree_print(parse_res)

#!/usr/bin/env python3

import sys
import ast
import re
from lark import Lark, Transformer, Tree
from lark.lexer import Lexer, Token
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, LexError, ParseError, VisitError
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
        self.pattern_str = terminal.pattern.to_regexp()
        self.pattern = re.compile(self.pattern_str)
        self.priority = terminal.priority

    def search(self, string, start_pos=0, end_pos=None):
        if end_pos is None:
            end_pos = len(string)
        return self.pattern.search(string, start_pos, end_pos)

    def match(self, string, start_pos=0, end_pos=None):
        if end_pos is None:
            end_pos = len(string)
        return self.pattern.match(string, start_pos, end_pos)

    def __str__(self):
        return "{}".format(self.pattern_str)

    def __repr__(self):
        return self.__str__()

class HognoseLexerError(LexError, UnexpectedInput):
    def __init__(self, message, seq, lex_pos, line, column, **kwargs):
        super().__init__()

        self.line = line
        self.column = column
        self.pos_in_stream = lex_pos
        self.message = message
        self.format_dict = kwargs

        if isinstance(seq, bytes):
            self.char = seq[lex_pos:lex_pos + 1].decode("ascii", "backslashreplace")
        else:
            self.char = seq[lex_pos]
        self._context = self.get_context(seq)

    def __str__(self):
        format_dict = self.format_dict
        if "{char}" in self.message:
            format_dict["char"] = self.char
        message = self.message.format(**format_dict)
        message += ", at line {line} col {col}".format(line=self.line, col=self.column)
        message += '\n\n' + self._context
        return message

class HognoseLexer(Lexer):
    special = {
        "ML_COMMENT_START": "multiline comment start",
        "ML_COMMENT_END": "multiline comment end"
    }

    def __init__(self, lexer_conf):
        print(lexer_conf.terminals_by_name)
        self.ignore = lexer_conf.ignore
        self.patterns = {}
        specials_needed = list(self.special.keys())
        for terminal in lexer_conf.terminals:
            terminal_name = terminal.name
            if terminal_name in self.special:
                if terminal_name in specials_needed:
                    specials_needed.remove(terminal_name)
                setattr(self, terminal_name.lower(), LexMatch(terminal))
            else:
                self.patterns[terminal.name] = LexMatch(terminal)
        if len(specials_needed):
            raise LexError("No {} specified".format(" ,".join(["{} symbol".format(self.special[x]) for x in specials_needed])))
        print(self.patterns)

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
                raise HognoseLexerError("Multiple equal length matches: {matches}", self.original_data,
                                        self.pos, self.line, self.col, matches=[(x.type, x.value) for x in lex_match])
            else:
                return lex_match[0]
        raise HognoseLexerError("Unknown symbol '{char}'", self.original_data, self.pos, self.line, self.col)

    def calc_new_pos_col_line(self, input_text, new_text_pos, old_text_pos=0, old_col=1, old_line=1):
        newline_count = input_text.count('\n', old_text_pos, new_text_pos)
        col = old_col
        line = old_line
        if newline_count != -1:
            line += newline_count
            col = new_text_pos - input_text.rfind('\n', old_text_pos, new_text_pos)
        else:
            col += new_text_pos - old_text_pos
        return col, line, new_text_pos

    def remove_ml_comment(self, input_text):
        ret_text = input_text
        ml_comment_stack = 0
        text_pos = 0
        col = 1
        line = 1
        last_opening_symbol_pos = None
        text_len = len(ret_text)
        while ml_comment_start_match := self.ml_comment_start.search(ret_text):
            col, line, text_pos = self.calc_new_pos_col_line(ret_text, ml_comment_start_match.start(), text_pos, col, line)
            last_opening_symbol_pos = (text_pos, line, col)
            while text_pos < text_len:
                advance_amount = 1
                if commenter_match := self.ml_comment_start.match(ret_text, text_pos):
                    ml_comment_stack += 1
                    advance_amount = commenter_match.end() - commenter_match.start()
                elif commenter_match := self.ml_comment_end.match(ret_text, text_pos):
                    ml_comment_stack -= 1
                    advance_amount = commenter_match.end() - commenter_match.start()

                replacement = "".join([ret_text[i] if ret_text[i].isspace() else ' ' for i in range(text_pos, text_pos + advance_amount)])
                ret_text = ret_text[:text_pos] + replacement + ret_text[text_pos + advance_amount:]
                col, line, text_pos = self.calc_new_pos_col_line(ret_text, text_pos + advance_amount, text_pos, col, line)

                if ml_comment_stack == 0:
                    break

            if ml_comment_stack != 0:
                raise HognoseLexerError("Unmatched multi-line comment opening symbol", self.original_data,
                                        last_opening_symbol_pos[0], last_opening_symbol_pos[1], last_opening_symbol_pos[2])

        if ml_comment_end_match := self.ml_comment_end.search(ret_text, text_pos):
            col, line, text_pos = self.calc_new_pos_col_line(ret_text, ml_comment_end_match.start(), text_pos, col, line)
            raise HognoseLexerError("Unmatched multi-line comment closing symbol", self.original_data, text_pos, col, line)
        return ret_text

    def preprocess(self, input_text):
        ret_text = input_text
        ret_text = self.remove_ml_comment(input_text)
        return ret_text

    def lex(self, data):
        self.original_data = data
        tokens = []
        self.col = 1
        self.line = 1
        self.pos = 0
        data = self.preprocess(data)
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
        if isinstance(tree, list) and len(tree) == 1:
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
    """,
    """
    a = 2e1 + 3e-5.1i - z
    b = "cheese \\" \\
    eez"
    c = r'''fries
    '''
    d = []
    e = [:]
    f = [
        1,
        'cheese'
    ]
    g = [
        b: a,
        c: f,

    ]
    """,
    """
    d = [];;;


    """,
    """
    # cheese
    d = [] #fries
    e = [:]
    # e = []
    """,
    """
    #* #* #* *# *#

    *#a=1#**#

    d=[]
    """,
    """
    cheese: a: fries = 1
    return to cheese
    """
]

#with open(sys.argv[1]) as f:
#    input_text = f.read()

if input_text is None:
    for input_text in input_texts:
        print("-"*80)
        for line_no, line in enumerate(input_text.splitlines()):
            print("{}: {}".format(line_no + 1, line))
        raw_parse_res = parser.parse(input_text)
        try:
            parse_res = HognosePostParse().transform(raw_parse_res)
        except VisitError as e:
            tree_print(raw_parse_res)
            raise e from None
        tree_print(parse_res)

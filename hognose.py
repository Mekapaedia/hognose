#!/usr/bin/env python3

import sys
import ast
import re
import copy
have_readline = False
try:
    import readline
    have_readline = True
except ImportError:
    have_readline = False
from cmd import Cmd
from lark import Lark, Tree
from lark.lexer import Lexer, Token
from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedInput, LexError, ParseError, VisitError
from lark.visitors import Transformer, Transformer_InPlace, Interpreter, v_args, Discard
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

def format_or_empty(string, nullable):
    return string.format(nullable) if nullable is not None else ""

class LexMatch:
    def __init__(self, terminal):
        self.pattern_str = terminal.pattern.to_regexp()
        if self.pattern_str.isalnum():
            self.pattern_str = r'\b' + self.pattern_str + r'\b'
        self.pattern = re.compile(self.pattern_str)
        self.priority = terminal.priority

    def search(self, string, start_pos=0, end_pos=None):
        return self.get_lex_match(string, start_pos, end_pos)

    def match(self, string, start_pos=0, end_pos=None):
        return self.get_lex_match(string, start_pos, end_pos, match=True)

    def get_lex_match(self, input_text, start_pos=0, end_pos=None, match=False):
        if end_pos is None:
            end_pos = len(input_text)
        match_func = self.pattern.match if match is True else self.pattern.search
        match = match_func(input_text, start_pos, end_pos)
        return match if match is None else match.group(0)

    def __str__(self):
        return "{}".format(self.pattern_str)

    def __repr__(self):
        return self.__str__()

class MLCommentMatch:
    def __init__(self, ml_comment_start, ml_comment_end):
        self.ml_comment_start_str = ml_comment_start.pattern.to_regexp()
        self.ml_comment_start = re.compile(self.ml_comment_start_str)
        self.ml_comment_end_str = ml_comment_end.pattern.to_regexp()
        self.ml_comment_end = re.compile(self.ml_comment_end_str)
        self.priority = max(ml_comment_start.priority, ml_comment_end.priority)

    def search(self, string, start_pos=0, end_pos=None):
        return self.get_ml_comment(string, start_pos, end_pos)

    def match(self, string, start_pos=0, end_pos=None):
        return self.get_ml_comment(string, start_pos, end_pos, match=True)

    def get_ml_comment(self, input_text, start_pos=0, end_pos=None, match=False):
        if end_pos is None:
            end_pos = len(input_text)
        ml_comment_start_func = self.ml_comment_start.match if match is True else self.ml_comment_start.search

        ml_comment_stack = 0
        text_pos = start_pos
        ret_match = None
        if ml_comment_start_match := ml_comment_start_func(input_text, text_pos, end_pos):
            ml_comment_stack += 1
            ret_match = ml_comment_start_match.group(0)
            text_pos = ml_comment_start_match.end()
            while ml_comment_stack > 0 and text_pos < end_pos:
                if commenter_match := self.ml_comment_start.match(input_text, text_pos, end_pos):
                    ml_comment_stack += 1
                    ret_match += commenter_match.group(0)
                    text_pos = commenter_match.end()
                elif commenter_match := self.ml_comment_end.match(input_text, text_pos, end_pos):
                    ml_comment_stack -= 1
                    ret_match += commenter_match.group(0)
                    text_pos = commenter_match.end()
                else:
                    ret_match += input_text[text_pos]
                    text_pos += 1

        if ml_comment_stack != 0:
            return None
        return ret_match

    def __str__(self):
        return "Multiline comment '{}' ... '{}'".format(self.ml_comment_start_str, self.ml_comment_end_str)

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
        self.ignore = lexer_conf.ignore
        self.patterns = {}
        specials_needed = list(self.special.keys())
        for terminal in lexer_conf.terminals:
            terminal_name = terminal.name
            if terminal_name in self.special:
                if terminal_name in specials_needed:
                    specials_needed.remove(terminal_name)
                setattr(self, terminal_name.lower(), terminal)
            else:
                self.patterns[terminal.name] = LexMatch(terminal)
        if len(specials_needed):
            raise LexError("No {} specified".format(" ,".join(["{} symbol".format(self.special[x]) for x in specials_needed])))
        self.patterns["ML_COMMENT"] = MLCommentMatch(self.ml_comment_start, self.ml_comment_end)
        self.ignore.append("ML_COMMENT")

    def get_token_match(self, data):
        matches = {}
        highest_prio = None
        for term_name, lex_match in self.patterns.items():
            match = lex_match.match(data)
            if match:
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

    def lex(self, data):
        self.original_data = data
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

    def _ambig_expr(self, tree):
        tighter_bound_group = [x for x in tree if x.children[1] is None]
        less_bound_group = [x for x in tree if x.children[1] is not None]
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
            raise ParseError("No ambiguity handler registered for {}\n\n{}".format(ambig_name, "\n\n".join([x.pretty() for x in tree])))
        return getattr(self, ambig_handler)(tree)

@v_args(tree=True)
class HognoseASTTransform(Transformer):

    def __init__(self, parser_rules, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newline_removes = {}
        for rule in parser_rules:
            rule_name = rule.origin.name.value
            rule_productions = [x.name for x in rule.expansion]
            rule_newlines = [i for i in range(0, len(rule_productions)) if rule_productions[i] == 'newlines']
            if rule_name not in self.newline_removes:
                self.newline_removes[rule_name] = {}
            self.newline_removes[rule_name][len(rule_productions)] = rule_newlines

    def __remove_newlines(self, name, tree):
        for del_index in sorted(self.newline_removes[name][len(tree)], reverse=True):
            del tree[del_index]
        return tree

    def __default__(self, name, children, meta):
        return Tree(name, self.__remove_newlines(name, children), meta)

    def properties(self, tree):
        del tree.children[1]
        more_properties = tree.children[1]
        del tree.children[1]
        if more_properties is not None:
            tree.children.extend(more_properties)
        return tree

    def more_properties(self, tree):
        del tree.children[1]
        more_properties = tree.children[1]
        del tree.children[1]
        if more_properties is not None:
            tree.children.extend(more_properties)
        return tree

    def terminated_expr(self, tree):
        if all([x is None for x in tree.children]):
            return Discard
        return tree.children[0]

    def expr_list(self, tree):
        original_child = tree.children[-1]
        del tree.children[-1]
        if original_child is not None:
            tree.children.extend(original_child.children)
        tree.children = [x for x in tree.children if x is not None]
        return tree

    def more_call_args(self, tree):
        del tree.children[0]
        del tree.children[0]
        del tree.children[0]
        original_child = tree.children[-1]
        del tree.children[-1]
        if original_child is not None:
            tree.children.extend(original_child.children)
        tree.children = [x for x in tree.children if x is not None]
        return tree

    def call_args_body(self, tree):
        del tree.children[-1]
        original_child = tree.children[-1]
        del tree.children[-1]
        if original_child is not None:
            tree.children.extend(original_child.children)
        tree.children = [x for x in tree.children if x is not None]
        return tree

    def call_args(self, tree):
        tree.children = [x for x in tree.children if x is not None]
        del tree.children[0]
        del tree.children[-1]
        ret_tree = tree
        if len(tree.children) > 0:
            ret_tree = tree.children[0]
        ret_tree.data = "call_args"
        return ret_tree

    def empty_expr(self, tree):
        return Discard

    def newlines(self, tree):
        return Discard

    def elexpr(self, tree):
        control_expr = tree.children[0]
        cond_body = control_expr.children[2]
        del control_expr.children[2]
        elexpr = cond_body.children[1]
        if elexpr is not None:
            cond_body.children[1] = cond_body.children[1].children[0]
        control_expr.children.extend(cond_body.children)
        return control_expr

    def control_exprs(self, tree):
        control_expr = tree.children[0]
        cond_body = control_expr.children[2]
        del control_expr.children[2]
        elexpr = cond_body.children[1]
        if elexpr is not None:
            cond_body.children[1] = cond_body.children[1].children[0]
        control_expr.children.extend(cond_body.children)
        return control_expr

    def more_param_eles(self, tree):
        del tree.children[0]
        del tree.children[0]
        original_child = tree.children[1]
        del tree.children[1]
        if original_child is not None:
            tree.children.extend(original_child.children)
        return tree

    def param_eles(self, tree):
        original_child = tree.children[1]
        del tree.children[1]
        if original_child is not None:
            tree.children.extend(original_child.children)
        return tree

    def param_ele(self, tree):
        if tree.children[2] is not None:
            tree.children[2] = tree.children[2].children[1]
        return tree

    def list(self, tree):
        del tree.children[0]
        del tree.children[-1]
        tree.children = [x for x in tree.children if x is not None]
        if len(tree.children) > 0 and tree.children[0] is not None:
            elements = tree.children[0]
            del tree.children[0]
            tree.children.extend(elements.children)
            tree.children = [x for x in tree.children if x is not None]
        return tree

    def list_body(self, tree):
        return tree.children[0]

    def list_elements(self, tree):
        more_elements = tree.children[1]
        del tree.children[1]
        if more_elements is not None and not isinstance(more_elements, Token):
            tree.children.extend(more_elements.children)
        return tree

    def more_list_elements(self, tree):
        if len(tree.children) == 1:
            return Discard
        del tree.children[0]
        del tree.children[0]
        next_element = tree.children[1]
        del tree.children[1]
        if next_element is not None and not isinstance(next_element, Token):
            tree.children.extend(next_element.children)
        return tree

    def dict(self, tree):
        del tree.children[0]
        del tree.children[-1]
        tree.children = [x for x in tree.children if x is not None]
        if len(tree.children) == 1:
            elements = tree.children[0]
            del tree.children[0]
            if not isinstance(elements, Token):
                tree.children.extend(elements.children)
        return tree

    def dict_element(self, tree):
        tree.children = [x for x in tree.children if x is not None]
        del tree.children[1]
        return tree

    def dict_elements(self, tree):
        more_elements = tree.children[1]
        del tree.children[1]
        if more_elements is not None:
            tree.children.extend(more_elements.children)
        return tree

    def more_dict_elements(self, tree):
        tree.children = [x for x in tree.children if x is not None]
        del tree.children[0]
        if len(tree.children) > 1:
            next_element = tree.children[1]
            del tree.children[1]
            if not isinstance(next_element, Token):
                tree.children.extend(next_element.children)
        return tree

    def group(self, tree):
        next_group = tree.children[1]
        del tree.children[1]
        if next_group is not None and not isinstance(next_group, Token):
            tree.children.extend(next_group.children)
        return tree

    def more_group(self, tree):
        del tree.children[0]
        del tree.children[0]
        del tree.children[0]
        next_group = tree.children[1]
        del tree.children[1]
        if next_group is not None:
            tree.children.extend(next_group.children)
        return tree

    def slice(self, tree):
        del tree.children[0]
        del tree.children[0]
        del tree.children[-1]
        del tree.children[-1]
        return tree

    def range(self, tree):
        del tree.children[2]
        del tree.children[2]
        del tree.children[2]
        return tree

    def range_start(self, tree):
        if tree.children[0] is not None:
            tree.children[0] = tree.children[0].children[0]
            if isinstance(tree.children[0], Token) and tree.children[0].type == "LT":
                tree.children[0] = None
        if tree.children[1] is not None:
            tree.children[1] = tree.children[1].children[0]
        return tree

    def range_end(self, tree):
        if tree.children[0] is not None:
            tree.children[0] = tree.children[0].children[0]
        if tree.children[1] is not None:
            tree.children[1] = tree.children[1].children[0]
            if isinstance(tree.children[1], Token) and tree.children[1].type == "GT":
                tree.children[1] = None
        return tree

    def range_step(self, tree):
        del tree.children[0]
        del tree.children[0]
        del tree.children[-2]
        return tree

    def for_in(self, tree):
        del tree.children[1]
        return tree

    def classdecl(self, tree):
        if len(tree.children) == 5 and tree.children[3] is None:
            del tree.children[3]
        return tree

    def class_parents(self, tree):
        del tree.children[0]
        del tree.children[0]
        del tree.children[0]
        more_class_parents = tree.children[1]
        del tree.children[1]
        if more_class_parents is not None:
            tree.children.extend(more_class_parents.children)
        return tree

    def more_class_parents(self, tree):
        del tree.children[0]
        del tree.children[0]
        del tree.children[0]
        more_class_parents = tree.children[1]
        del tree.children[1]
        if more_class_parents is not None:
            tree.children.extend(more_class_parents.children)
        return tree

    def class_parent_ele(self, tree):
        return tree

    def more_disjunction(self, tree):
        del tree.children[0]
        del tree.children[1]
        return tree

    def more_conjunction(self, tree):
        del tree.children[0]
        del tree.children[1]
        return tree

class MissingSymbolError(Exception):
    pass

class Scope:
    def __init__(self, parent_scope=None,
                       loop_scope=False,
                       function_scope=False,
                       symbols=None,
                       capture_scope=None,
                       captured_scope=None,
                       properties=None):
        self.parent_scope = parent_scope
        self.loop_scope = loop_scope
        self.function_scope = function_scope
        self.symbols = symbols if symbols is not None else {}
        self.break_called = False
        self.break_val = None
        self.defer_exprs = []
        self.capture_scope = capture_scope
        self.captured_scope = captured_scope
        if self.capture_scope is None:
            if self.captured_scope is not None:
                self.capture_scope = True
            elif self.parent_scope is not None:
                self.capture_scope = self.parent_scope.capture_scope
            else:
                self.capture_scope = False
        if self.capture_scope:
            for value in self.symbols.values():
                if hasattr(value, "captured_scope"):
                    value.captured_scope = self
        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
        for symbol in self.symbols:
            if symbol not in self.properties:
                self.properties[symbol] = []

    def get(self, symbol, immediate=False):
        if symbol in self.symbols:
            return self.symbols[symbol]
        elif immediate is False and self.captured_scope is not None and self.captured_scope.has(symbol):
            return self.captured_scope.get(symbol)
        elif immediate is False and self.parent_scope is not None and self.parent_scope.has(symbol):
            return self.parent_scope.get(symbol)
        raise MissingSymbolError("No symbol '{}'".format(symbol))

    def get_props(self, symbol, immediate=False):
        if symbol in self.properties:
            return self.properties[symbol]
        elif immediate is False and self.captured_scope is not None and self.captured_scope.has(symbol):
            return self.captured_scope.get_properties(symbol)
        elif immediate is False and self.parent_scope is not None and self.parent_scope.has(symbol):
            return self.parent_scope.get_properties(symbol)
        raise MissingSymbolError("No symbol '{}'".format(symbol))

    def assign(self, symbol, value, immediate=False, properties=None):
        if symbol in self.symbols or immediate:
            pass
        elif self.captured_scope is not None and self.captured_scope.has(symbol):
            return self.captured_scope.assign(symbol, value)
        elif self.parent_scope is not None and self.parent_scope.has(symbol):
            return self.parent_scope.assign(symbol, value)
        if self.capture_scope and hasattr(value, "captured_scope"):
            if value.captured_scope is None:
                value.captured_scope = self
        self.symbols[symbol] = value
        if properties is not None:
            self.properties[symbol] = properties
        elif symbol not in self.properties:
            self.properties[symbol] = set()
        return self.symbols[symbol]

    def has(self, symbol):
        if symbol in self.symbols:
            return True
        elif self.captured_scope is not None and self.captured_scope.has(symbol):
            return True
        elif self.parent_scope is not None and self.parent_scope.has(symbol):
            return True
        return False

    def call_break(self, loop_break=False, function_break=False, loop_continue=False, break_val=None):
        self.break_val = break_val
        if not (loop_continue is True and self.loop_scope is True):
            self.break_called = True
        if self.parent_scope is not None:
            if (loop_break or loop_continue) and self.loop_scope is False:
                self.parent_scope.call_break(loop_break=loop_break, function_break=function_break, break_val=break_val)
            elif function_break and self.function_scope is False:
                self.parent_scope.call_break(loop_break=loop_break, function_break=function_break, break_val=break_val)

    def add_defer(self, defer_expr):
        self.defer_exprs.insert(0, defer_expr)

    def has_defers(self):
        return len(self.defer_exprs) > 0

    def run_defers(self):
        last_res = None
        for defer_expr in self.defer_exprs:
            last_res = defer_expr.eval(self)
        return last_res

    def push_scope(self, loop_scope=False,
                         function_scope=False,
                         symbols=None,
                         captured_scope=None):
        return Scope(parent_scope=self,
                     loop_scope=loop_scope,
                     function_scope=function_scope,
                     symbols=symbols,
                     captured_scope=captured_scope)

    def pop_scope(self):
        if self.parent_scope is None:
            raise ValueError("Cannot pop top scope")
        return self.parent_scope

    def copy(self, parent_scope=None, loop_scope=None, function_scope=None, symbols=None):
        if symbols is None:
            symbols = {**self.symbols}
        else:
            symbols = {**self.symbols, **symbols}
        if parent_scope is None:
            parent_scope = self.parent_scope
        if loop_scope is None:
            loop_scope = self.loop_scope
        if function_scope is None:
            function_scope = self.function_scope
        properties = self.properties # FIXME
        return Scope(parent_scope=parent_scope, loop_scope=loop_scope, function_scope=function_scope, symbols=symbols, properties=properties)

    def split_by_prop(self, properties, parent_scope=None, loop_scope=None, function_scope=None):
        return (self.get_by_prop(properties, parent_scope=parent_scope, loop_scope=loop_scope, function_scope=function_scope),
                self.get_by_prop(properties, parent_scope=parent_scope, loop_scope=loop_scope, function_scope=function_scope, inverse=True))

    def get_by_prop(self, properties, parent_scope=None, loop_scope=None, function_scope=None, inverse=False):
        if parent_scope is None:
            parent_scope = self.parent_scope
        if loop_scope is None:
            loop_scope = self.loop_scope
        if function_scope is None:
            function_scope = self.function_scope
        if isinstance(properties, list):
            properties = set(properties)
        new_symbols = {}
        new_properties = {}
        for symbol in self.symbols:
            props_match = inverse
            if isinstance(properties, set) and self.properties[symbol] == properties:
                props_match = not inverse
            elif isinstance(properties, str) and properties in self.properties[symbol]:
                props_match = not inverse
            if props_match:
                new_symbols[symbol] = self.symbols[symbol]
                new_properties[symbol] = self.properties[symbol]
        return Scope(parent_scope=parent_scope, loop_scope=loop_scope, function_scope=function_scope, symbols=new_symbols, properties=new_properties)

    def symbol_names(self):
        symbol_names = list(self.symbols.keys())
        if self.captured_scope is not None:
            symbol_names.extend(self.captured_scope.symbol_names())
        if self.parent_scope is not None:
            symbol_names.extend(self.parent_scope.symbol_names())
        return symbol_names

class LiteralTrue:
    def __init__(self):
        pass

    def __str__(self):
        return "LiteralTrue"

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return True

class LiteralFalse:
    def __init__(self):
        pass

    def __str__(self):
        return "LiteralFalse"

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return False

class LiteralString:
    def __init__(self, value):
        self.value = str(value)

    def __str__(self):
        return "LiteralString: '{}'".format(self.value)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return self.value

class LiteralFloat:
    def __init__(self, value):
        self.value = float(value)

    def __str__(self):
        return "LiteralFloat: {}".format(self.value)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return self.value

class LiteralInt:
    def __init__(self, value):
        self.value = int(value)

    def __str__(self):
        return "LiteralInt: {}".format(self.value)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return self.value

class LiteralList:
    def __init__(self, elements):
        self.elements = list(elements) if elements is not None else []

    def __str__(self):
        return "LiteralList: {}".format(self.elements)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return [x.eval(symbol_table) for x in self.elements]

class DictEle:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __str__(self):
        return "DictEle: {}:{}".format(self.key, self.value)

    def __repr__(self):
        return self.__str__()

class LiteralDict:
    def __init__(self, elements):
        self.elements = dict(elements) if elements is not None else {}

    def __str__(self):
        return "LiteralDict: {}".format(self.elements)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return {k.eval(symbol_table): v.eval(symbol_table) for k, v in self.elements.items()}

class Group:
    def __init__(self, exprs):
        self.exprs = list(exprs)

    def __str__(self):
        return "Group: {}".format(", ".join([str(x) for x in self.exprs]))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return [x.eval(symbol_table) for x in self.exprs]

class Range:
    def __init__(self, start, end, step, start_closed=False, end_closed=False):
        self.start = start
        self.end = end
        self.step = step
        self.start_closed = start_closed
        self.end_closed = end_closed

    def to_list(self):
        start = self.start
        if start is None:
            start = 0 # FIXME
        step = self.step
        end = self.end
        if step is None:
            step = 1
        if start is None:
            start = 0
        if end is None:
            end = 0
            if step < 0:
                end = None if self.end_closed is False else 0
        if self.start_closed is True:
            start += step
        if self.end_closed is True:
            end -= step
        return list(range(start, end + 1 if step > 0 else end, step))

class RangeExpr:
    def __init__(self, start, end, start_closed=False, end_closed=False, step=None):
        self.start = start
        self.end = end
        self.start_closed = start_closed
        self.end_closed = end_closed
        self.step = step

    def __str__(self):
        return "Range: start{}:{} stop{}:{} step:{}".format(
            self.start, "(closed)" if self.start_closed else "",
            self.end, "(closed)" if self.end_closed else "",
            self.step)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        start = self.start.eval(symbol_table) if self.start is not None else None
        step = self.step.eval(symbol_table) if self.step is not None else None
        end = self.end.eval(symbol_table) if self.end is not None else None
        return Range(start, end, step, start_closed=self.start_closed, end_closed=self.end_closed)

class SliceExpr:
    def __init__(self, target, slice):
        self.target = target
        self.slice = slice

    def __str__(self):
        return "SliceExpr: {}({})".format(self.target, self.slice)

    def __repr__(self):
        return self.__str__()

    def get_slice_eles(self, target, slice_eles):
        start = slice_eles.start
        step = slice_eles.step
        end = slice_eles.end
        if step is None:
            step = 1
        if start is None:
            start = 0 if slice_eles.start_closed is False else step
        if end is None:
            end = len(target)
            if step < 0:
                end = None if slice_eles.end_closed is False else 0
            elif slice_eles.end_closed is True:
                end -= step
        return start, step, end

    def handle_assign(self, target, slice_ele, assign_val):
        if isinstance(slice_ele, Range):
            start, step, end = self.get_slice_eles(target, slice_ele)
            target[start:end:step] = assign_val
        else:
            target[slice_ele] = assign_val
        return target

    def handle_slice(self, target, slice_ele):
        if isinstance(slice_ele, Range):
            start, step, end = self.get_slice_eles(target, slice_ele)
            return target[start:end:step]
        else:
            return target[slice_ele]

    def eval(self, symbol_table, assign=False, assign_val=None):
        target = self.target.eval(symbol_table)
        slice_eles = self.slice.eval(symbol_table)
        ret_val = None
        if isinstance(slice_eles, list):
            if assign:
                for val_ele, slice_ele in enumerate(slice_eles):
                    val = assign_val
                    if isinstance(assign_val, list) and len(assign_val) == len(slice_eles):
                        val = assign_val[val_ele]
                    target = self.handle_assign(target, slice_ele, val)
            ret_val = [self.handle_slice(target, x) for x in slice_eles]
        else:
            if assign:
                target = self.handle_assign(target, slice_eles, assign_val)
            ret_val = self.handle_slice(target, slice_eles)
        return ret_val

class Slice:
    def __init__(self, slice):
        self.slice = slice

    def __str__(self):
        return "Slice: {}".format(self.slice)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return self.slice.eval(symbol_table)

class FieldAccessExpr:
    def __init__(self, target, field):
        self.target = target
        self.field = field

    def __str__(self):
        return "FieldExpr: {}.{}".format(self.target, self.field)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, assign=False, assign_val=None):
        target = self.target.eval(symbol_table)
        field_name = self.field.eval(symbol_table)
        if assign:
            target = target.assign_field(field_name, assign_val)
        return target.get_field(field_name)

class FieldAccess:
    def __init__(self, field_access):
        self.field_access = field_access

    def __str__(self):
        return "FieldAccess: {}".format(self.field_access)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return self.field_access.name

class NameRef:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "NameRef: {}".format(self.name)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return symbol_table.get(self.name)

class PosArg:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "PosArg: {}".format(str(self.value))

    def __repr__(self):
        return self.__str__()

class KwArg:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return "KwArg: {}={}".format(self.name, str(self.value))

    def __repr__(self):
        return self.__str__()

class CallArgs:
    def __init__(self, *pos_args, **kw_args):
        self.pos_args = list(pos_args)
        self.kw_args = kw_args

    def add_arg(self, new_arg):
        if isinstance(new_arg, PosArg):
            self.pos_args.append(new_arg.value)
        elif isinstance(new_arg, KwArg):
            self.kw_args[new_arg.name] = new_arg.value
        else:
            raise ValueError("Unknown arg")

    def __str__(self):
        return "{}".format(", ".join([str(x) for x in self.pos_args] + ["{}={}".format(k, v) for k, v in self.kw_args.items()]))

    def __repr__(self):
        return self.__str__()

class FuncCall:
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

    def __str__(self):
        return "FuncCall: {}({})".format(self.callee, self.args)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return self.callee.eval(symbol_table).eval(symbol_table, self.args.pos_args, self.args.kw_args)

class BinOp:
    def __init__(self, lhs, operator, rhs):
        self.lhs = lhs
        self.operator = operator
        self.rhs = rhs

    def __str__(self):
        return "BinOp: ({} {} {})".format(self.lhs, self.operator, self.rhs)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        lhs_val = self.lhs.eval(symbol_table)
        if self.operator == "or" or self.operator == "||":
            if lhs_val is True:
                return lhs_val
            else:
                return self.rhs.eval(symbol_table)
        elif self.operator == "and" or self.operator == "&&":
            if lhs_val is False:
                return lhs_val
            else:
                return self.rhs.eval(symbol_table)
        rhs_val = self.rhs.eval(symbol_table)
        if self.operator == "+":
            return lhs_val + rhs_val
        elif self.operator == "-":
            return lhs_val - rhs_val
        elif self.operator == "*":
            return lhs_val * rhs_val
        elif self.operator == "/":
            return lhs_val / rhs_val
        elif self.operator == "//":
            return lhs_val // rhs_val
        elif self.operator == "%":
            return lhs_val % rhs_val
        elif self.operator == ">":
            return lhs_val > rhs_val
        elif self.operator == ">=":
            return lhs_val >= rhs_val
        elif self.operator == "<":
            return lhs_val < rhs_val
        elif self.operator == "<=":
            return lhs_val <= rhs_val
        elif self.operator == "==":
            return lhs_val == rhs_val
        elif self.operator == "!=":
            return lhs_val != rhs_val
        elif self.operator == ">>":
            return lhs_val >> rhs_val
        elif self.operator == "<<":
            return lhs_val << rhs_val
        elif self.operator == "&":
            return lhs_val & rhs_val
        elif self.operator == "^":
            return lhs_val ^ rhs_val
        elif self.operator == "|":
            return lhs_val | rhs_val
        elif self.operator == "**":
            return lhs_val ** rhs_val
        else:
            raise ValueError("Unhandled operator '{}'".format(self.operator))

class UnOp:
    def __init__(self, operator, rhs):
        self.operator = operator
        self.rhs = rhs

    def __str__(self):
        return "UnOp: ({} {})".format(self.operator, self.rhs)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        rhs_val = self.rhs.eval(symbol_table)
        if self.operator == "+":
            return +rhs_val
        elif self.operator == "-":
            return -rhs_val
        elif self.operator == "~":
            return ~rhs_val
        elif self.operator == "not" or self.operator == "!":
            return not rhs_val
        else:
            raise ValueError("Unhandled operator '{}'".format(self.operator))

class ExitExpr:
    def __init__(self, exit_type, exit_val, exit_dir):
        self.exit_type = exit_type
        self.exit_val = exit_val
        self.exit_dir = exit_dir

    def __str__(self):
        return "Exit: {}{}{}".format(self.exit_type,
               " ({})".format(self.exit_val) if self.exit_val is not None else "",
               " to {}" if self.exit_dir is not None else "")

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        if self.exit_type == "defer":
            symbol_table.add_defer(self.exit_val)
        else:
            loop_break = self.exit_type == "break"
            loop_continue = self.exit_type == "continue"
            function_break = self.exit_type == "return" or self.exit_type == "yield"
            break_val = self.exit_val.eval(symbol_table) if self.exit_val is not None else None
            symbol_table.call_break(loop_break=loop_break, loop_continue=loop_continue,
                                    function_break=function_break, break_val=break_val)
            return break_val

class AssignOp:
    def __init__(self, target, type_expr, operator, value):
        self.target = target
        self.type_expr = type_expr
        self.operator = operator
        self.value = value

    def __str__(self):
        return "Assign: {} {} {} {}".format(self.target, self.type_expr, self.operator, self.value)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, properties=None, class_decl=False):
        target = self.target
        # TODO!!!
        if isinstance(target, NameRef):
            return symbol_table.assign(target.name, self.value.eval(symbol_table), properties=properties, immediate=class_decl)
        elif class_decl is True:
            raise Exception("class_decl is true but it doesn't make sense")
        elif isinstance(target, SliceExpr):
            return target.eval(symbol_table, assign=True, assign_val=self.value.eval(symbol_table))
        elif isinstance(target, FieldAccessExpr):
            return target.eval(symbol_table, assign=True, assign_val=self.value.eval(symbol_table))

class IfExpr:
    def __init__(self, guard, body, else_expr):
        self.guard = guard
        self.body = body
        self.else_expr = else_expr

    def __str__(self):
        return "If: {} then {}{}".format(self.guard, self.body,
                " else {}".format(self.else_expr) if self.else_expr is not None else "")

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        symbol_table = symbol_table.push_scope()
        guard_res = self.guard.eval(symbol_table)
        if guard_res is True:
            val = self.body.eval(symbol_table)
            if symbol_table.break_called and symbol_table.break_val is not None:
                val = symbol_table.break_val
            return val
        symbol_table = symbol_table.pop_scope()
        if self.else_expr is not None:
            return self.else_expr.eval(symbol_table)

class WhileExpr:
    def __init__(self, guard, body, else_expr):
        self.guard = guard
        self.body = body
        self.else_expr = else_expr

    def __str__(self):
        return "If: {} then {}{}".format(self.guard, self.body,
                " else {}".format(self.else_expr) if self.else_expr is not None else "")

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        at_least_once = False
        last_val = None
        symbol_table = symbol_table.push_scope(loop_scope=True)
        while self.guard.eval(symbol_table) is True:
            at_least_once = True
            last_val = self.body.eval(symbol_table)
            if symbol_table.break_called:
                if symbol_table.break_val is not None:
                    last_val = symbol_table.break_val
                break
        symbol_table = symbol_table.pop_scope()
        if at_least_once:
            return last_val
        elif self.else_expr is not None:
            return self.else_expr.eval(symbol_table)

class ArgumentError(Exception):
    pass

class FnDef:
    def __init__(self, body, pos_args_names=None,
                             args=None,
                             va_args_name=None,
                             va_kw_args_name=None,
                             type_expr=None,
                             captured_scope=None):
        self.body = body
        self.type_expr = type_expr
        self.pos_args_names = pos_args_names if pos_args_names is not None else []
        self.args = args if args is not None else {}
        self.va_args_name = va_args_name
        self.va_kw_args_name = va_kw_args_name
        self.captured_scope = captured_scope

    def __str__(self):
        return "FnDef: ({}) {{{}}}".format(format_or_empty("{}", self.args), self.body)

    def __repr__(self):
        return self.__str__()

    def eval_args(self, symbol_table, pos_args, kw_args):
        if pos_args is None:
            pos_args = []
        if kw_args is None:
            kw_args = {}
        new_args = {}
        if len(pos_args) > len(self.pos_args_names) and self.va_args_name is None:
            raise ArgumentError("Function takes a maximum of {} positional arguments, not {}".format(len(self.pos_args_names), len(pos_args)))
        if self.va_args_name is not None:
            new_args[self.va_args_name] = []
        for pos_arg_num, pos_arg in enumerate(pos_args):
            arg_val = pos_arg.eval(symbol_table)
            if pos_arg_num >= len(self.pos_args_names):
                new_args[self.va_args_name].append(arg_val)
            else:
                new_args[self.pos_args_names[pos_arg_num]] = arg_val
        if self.va_kw_args_name is not None:
            new_args[self.va_kw_args_name] = {}
        for arg_name, arg_val in kw_args.items():
            arg_val = arg_val.eval(symbol_table)
            if arg_name in self.args:
                if arg_name in new_args:
                    raise ArgumentError("Multiple definitions for argument '{}'".format(arg_name))
                elif self.args[arg_name].pos_only is True:
                    raise ArgumentError("Argument '{}' is positional-only".format(arg_name))
            elif self.va_kw_args_name is not None:
                new_args[self.va_kw_args_name][arg_name] = arg_val
            else:
                raise ArgumentError("Unknown arg name '{}'".format(arg_name))
        for arg_name, arg_val in self.args.items():
            if arg_name not in new_args:
                if self.args[arg_name].default is not None:
                    new_args[arg_name] = self.args[arg_name].default.eval(symbol_table)
                else:
                    raise ArgumentError("Function missing value for argument '{}'".format(arg_name))
        return symbol_table.push_scope(function_scope=True,
                                       symbols=new_args,
                                       captured_scope=self.captured_scope)

    def eval(self, symbol_table, pos_args=None, kw_args=None):
        return self.body.eval(self.eval_args(symbol_table, pos_args, kw_args))

class BuiltinFnDef(FnDef):
    def __init__(self, builtinfn, **kwargs):
        super().__init__(builtinfn, **kwargs)

    def eval(self, symbol_table, pos_args=None, kw_args=None):
        return self.body(*[x.eval(symbol_table) for x in pos_args], **{k: v.eval(symbol_table) for k, v in kw_args.items()})

    def __str__(self):
        return "BuiltinFnDef: {}".format(self.body)

class DeclArg:
    def __init__(self, name, default=None, type_expr=None, pos_only=False, kw_only=False):
        self.name = name
        self.default = default
        self.pos_only = pos_only
        self.kw_only = kw_only
        self.type_expr = type_expr

    def __str__(self):
        return "DeclArg: {}{}".format(self.name, format_or_empty(" ={}", self.default))

    def __repr__(self):
        return self.__str__()

class VaDeclArg:
    def __init__(self, name, type_expr=None):
        self.name = name
        self.type_expr = type_expr

    def __str__(self):
        return "VaDeclArg: {}".format(self.name)

    def __repr__(self):
        return self.__str__()

class KwVaDeclArg:
    def __init__(self, name, type_expr=None):
        self.name = name
        self.type_expr = type_expr

    def __str__(self):
        return "KwVaDeclArg: {}".format(self.name)

    def __repr__(self):
        return self.__str__()

class DeclArgList:
    def __init__(self, args):
        self.args = args if args is not None else []

    def __str__(self):
        return "DeclArgList: {}".format(", ".join([str(x) for x in self.args]))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, class_decl=False):
        pos_args_names = [x.name.name for x in self.args if x.kw_only is False]
        args_dict = {v.name.name: v for v in self.args}
        va_args_name = None
        va_kw_args_name = None
        return pos_args_names, args_dict, va_args_name, va_kw_args_name

class FnDeclExpr:
    def __init__(self, name, args, body, type_expr):
        self.name = name
        self.args = args
        self.body = body
        self.type_expr = type_expr

    def __str__(self):
        return "FnDecl: {}({}) {}".format(self.name if self.name is not None else "", self.args if self.args is not None else "", self.body)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, properties=None, class_decl=False):
        pos_args_names, args_dict, va_args_name, va_kw_args_name = self.args.eval(symbol_table)
        fndef = FnDef(self.body, pos_args_names=pos_args_names, args=args_dict, va_args_name=va_args_name, va_kw_args_name=va_kw_args_name, type_expr=self.type_expr)
        if self.name:
            symbol_table.assign(self.name.name, fndef, properties=properties, immediate=class_decl)
        return fndef

class ForExpr:
    def __init__(self, induction_var, induction_expr, body, else_expr=None):
        self.induction_var = induction_var
        self.induction_expr = induction_expr
        self.body = body
        self.else_expr = else_expr

    def __str__(self):
        return "For: {} in {} do {}".format(self.induction_var,
                                            self.induction_expr,
                                            self.body)
    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, class_decl=False):
        at_least_once = False
        last_val = None
        induction_var = self.induction_var.expr.name # FIXME
        symbol_table = symbol_table.push_scope(loop_scope=True)
        induction_list = self.induction_expr.eval(symbol_table)
        if isinstance(induction_list, Range): # FIXME
            induction_list = induction_list.to_list()
        for ele in induction_list:
            at_least_once = True
            symbol_table.assign(induction_var, ele, immediate=True)
            last_val = self.body.eval(symbol_table)
            if symbol_table.break_called:
                if symbol_table.break_val is not None:
                    last_val = symbol_table.break_val
                break
        symbol_table = symbol_table.pop_scope()
        if at_least_once:
            return last_val
        elif self.else_expr is not None:
            return self.else_expr.eval(symbol_table)

class DefaultInit:
    def __init__(self):
        pass

    def __str__(self):
        return "DefaultInit:"

    def __repr__(self):
        return self.__str__()

class ObjDef:
    def __init__(self, obj_type, members=None, instance_members=None, parents=None, obj_class=None, obj_name=None, pos_args=None, kw_args=None, symbol_table=None):
        self.obj_type = obj_type
        self.obj_class = obj_class
        self.obj_name = obj_name if obj_name is not None else "Anonymous"
        self.members = members if members is not None else Scope()
        self.instance_members = instance_members if instance_members is not None else Scope()
        self.parents = parents
        self.captured_scope = None
        new_members = {}
        new_properties = {}
        new_instance_properties = {}
        new_instance_members = {}
        if self.parents is not None:
            for parent in self.parents:
                for symbol_name, obj in parent.members.symbols.items():
                    new_members[symbol_name] = copy.copy(obj)
                    new_properties[symbol_name] = parent.members.get_props(symbol_name)
                for symbol_name, obj in parent.instance_members.symbols.items():
                    new_instance_members[symbol_name] = obj
                    new_instance_properties[symbol_name] = parent.instance_members.get_props(symbol_name)

        if self.obj_class is not None:
            if self.obj_type == "object":
                for symbol_name, obj in self.obj_class.instance_members.symbols.items():
                    new_members[symbol_name] = obj
                    new_properties[symbol_name] = self.obj_class.instance_members.get_props(symbol_name)

        for symbol_name, obj in self.members.symbols.items():
            new_members[symbol_name] = obj
            new_properties[symbol_name] = self.members.get_props(symbol_name)
        for symbol_name, obj in self.instance_members.symbols.items():
            new_instance_members[symbol_name] = obj
            new_instance_properties[symbol_name] = self.instance_members.get_props(symbol_name)

        self.members = Scope(symbols=new_members, properties=new_properties, capture_scope=True, parent_scope=symbol_table)
        self.instance_members = Scope(symbols=new_instance_members, properties=new_instance_properties, capture_scope=True, parent_scope=symbol_table)
        if self.obj_class is not None:
            self.members.assign("cls", self.obj_class, immediate=True)
        self.members.assign("self", self, immediate=True)

        obj_init = None
        if self.members.has("init"):
            obj_init = self.members.get("init")

        if obj_init is None or isinstance(obj_init, DefaultInit):
            pos_args = pos_args if pos_args is not None else []
            kw_args = kw_args if kw_args is not None else {}
            set_members = []
            pos_members_names = [x for x in self.members.symbols.keys() if x not in ["self", "cls", "init"]]
            if len(pos_args) > len(pos_members_names):
                raise ArgumentError("Object only takes {} arguments".format(len(pos_args)))
            for pos_arg_ele, pos_arg in enumerate(pos_args):
                pos_arg_name = pos_members_names[pos_arg_ele]
                self.members.assign(pos_arg_name, pos_arg.eval(symbol_table), immediate=True)
                set_members.append(pos_arg_name)
            for arg_name, arg_val in kw_args.items():
                if arg_name in set_members:
                    raise ArgumentError("Multiple values for arg '{}'".format(arg_name))
                self.members.assign(arg_name, arg_val.eval(symbol_table), immediate=True)
                set_members.append(arg_name)
            self.members.assign("init", DefaultInit(), immediate=True)
        else:
            obj_init.eval(symbol_table, pos_args=pos_args, kw_args=kw_args)

    def __str__(self):
        return "ObjDef: {} {} {}".format(self.obj_type, self.obj_name, [str(x) for x in self.parents] if self.parents is not None else "")

    def __repr__(self):
        return self.__str__()

    def assign_field(self, field_name, assign_val):
        self.members.assign(field_name, assign_val, immediate=True)
        return self

    def get_field(self, field_name):
        return self.members.get(field_name, immediate=True)

    def eval(self, symbol_table, pos_args=None, kw_args=None, class_decl=False):
        if self.obj_type == "class":
            return ObjDef("object", obj_class=self, symbol_table=symbol_table, pos_args=pos_args, kw_args=kw_args)
        else:
            raise ValueError("Not class")

class ClassDecl:
    def __init__(self, class_type, name=None, body=None, parents=None):
        self.class_type = class_type
        self.name = name
        self.body = body
        self.parents = parents if parents is not None else []

    def __str__(self):
        return "ClassDecl: {}{}{}{}".format(self.class_type,
                                          format_or_empty(" {}", self.name),
                                          format_or_empty(" {}", self.body),
                                          format_or_empty(": {}", self.parents))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, properties=None, class_decl=False):
        members = self.body.eval(symbol_table, class_decl=True)
        class_members, instance_members = members.split_by_prop("static")
        parents = [x.eval(symbol_table) for x in self.parents]
        classdef = ObjDef(self.class_type, members=class_members, instance_members=instance_members, parents=parents, obj_name=self.name.name, symbol_table=symbol_table)
        if self.name:
            symbol_table.assign(self.name.name, classdef, properties=properties, immediate=class_decl)
        return classdef

class ExprList:
    def __init__(self, exprs):
        self.exprs = list(exprs)

    def __str__(self):
        return "ExprList: {}".format("\n".join([str(x) for x in self.exprs]))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, class_decl=False, return_scope=False):
        last_res = None
        symbol_table = symbol_table.push_scope()
        if class_decl is True:
            return_scope = True
        for expr in self.exprs:
            next_res = None
            if class_decl is True:
                next_res = expr.eval(symbol_table, class_decl=class_decl)
            else:
                next_res = expr.eval(symbol_table)
            if symbol_table.break_called:
                if symbol_table.break_val is not None:
                    last_res = symbol_table.break_val
                elif next_res is not None:
                    last_res = next_res
                break
            last_res = next_res
        if return_scope:
            return symbol_table
        return last_res

class Expr:
    def __init__(self, expr, properties=None):
        self.expr = expr
        self.properties = properties

    def __str__(self):
        return "{}{}".format(format_or_empty("{} ", self.properties), self.expr)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, class_decl=False, return_scope=False):
        ret_val = None
        if class_decl is True:
            return_scope = True
        if self.properties is not None or class_decl is not False: # FIXME
            ret_val = self.expr.eval(symbol_table, properties=self.properties, class_decl=class_decl)
        else:
            ret_val = self.expr.eval(symbol_table)
        if symbol_table.break_called:
            if symbol_table.break_val is not None:
                ret_val = symbol_table.break_val
        if return_scope:
            return symbol_table
        return ret_val

class ConjRhs:
    def __init__(self, operator, rhs):
        self.operator = operator
        self.rhs = rhs

class DisjRhs:
    def __init__(self, operator, rhs):
        self.operator = operator
        self.rhs = rhs

class HognoseASTGen(Interpreter):
    def visit(self, tree, **kwargs):
        return getattr(self, tree.data)(tree, **kwargs)

    def visit_or_default(self, tree, default, **kwargs):
        if tree is None:
            return default
        elif isinstance(tree, Tree):
            return self.visit(tree, **kwargs)
        elif isinstance(tree, Token):
            return tree.value
        else:
            return tree

    def visit_or_none(self, tree, **kwargs):
        return self.visit_or_default(tree, None, **kwargs)

    def __default__(self, tree):
        name = tree.data if hasattr(tree, "data") else tree.value
        raise ValueError("No handler for '{}'".format(name))

    def parens(self, tree):
        return self.visit(tree.children[1])

    def bin(self, tree):
        return LiteralInt(int(tree.children[0].value, 2))

    def oct(self, tree):
        return LiteralInt(int(tree.children[0].value, 8))

    def hex(self, tree):
        return LiteralInt(int(tree.children[0].value, 16))

    def decimal(self, tree):
        str_val = tree.children[0].value
        if "." in str_val:
            return LiteralFloat(float(str_val))
        else:
            return LiteralInt(int(str_val))

    def d_sl_string(self, tree):
        new_string = tree.children[0].value[1:-1]
        return LiteralString(str(new_string))

    def s_sl_string(self, tree):
        new_string = tree.children[0].value[1:-1]
        return LiteralString(str(new_string))

    def string(self, tree):
        return self.visit(tree.children[0])

    def true(self, tree):
        return LiteralTrue()

    def false(self, tree):
        return LiteralFalse()

    def name(self, tree):
        return NameRef(tree.children[0].value)

    def self_ref(self, tree):
        return NameRef(tree.children[0].value)

    def cls_ref(self, tree):
        return NameRef(tree.children[0].value)

    def list(self, tree):
        return LiteralList([self.visit(x) for x in tree.children])

    def dict_element(self, tree):
        return DictEle(self.visit(tree.children[0]), self.visit(tree.children[-1]))

    def dict(self, tree):
        return LiteralDict({v.key: v.value for v in [self.visit(x) for x in tree.children]})

    def range_end(self, tree):
        end = self.visit_or_none(tree.children[1])
        end_closed = False if tree.children[0] is None else True
        return end, end_closed

    def range_step(self, tree):
        return self.visit(tree.children[0])

    def range_start(self, tree):
        start = self.visit_or_none(tree.children[0])
        start_closed = False if tree.children[1] is None else True
        return start, start_closed

    def range(self, tree):
        start, start_closed = self.visit(tree.children[0])
        step = self.visit_or_none(tree.children[1])
        end, end_closed = self.visit(tree.children[2])
        return RangeExpr(start=start, start_closed=start_closed,
                         end=end, end_closed=end_closed, step=step)

    def field_access(self, tree):
        return FieldAccess(self.visit(tree.children[1]))

    def slice(self, tree):
        return Slice(self.visit(tree.children[0]))

    def pos_arg(self, tree):
        return PosArg(self.visit(tree.children[0]))

    def kw_arg(self, tree):
        return KwArg(self.visit(tree.children[0]).name, self.visit(tree.children[2]))

    def call_args(self, tree):
        ret_args = CallArgs()
        for child in tree.children:
            ret_args.add_arg(self.visit(child))
        return ret_args

    def primary_assign(self, tree):
        ref = self.visit(tree.children[0])
        access = None
        if len(tree.children) > 1:
            access = self.visit(tree.children[1])
        else:
            return ref
        if isinstance(access, Slice):
            return SliceExpr(ref, access)
        elif isinstance(access, FieldAccess):
            return FieldAccessExpr(ref, access)
        else:
            raise ValueError("Haven't handled other stuff yet")

    def primary(self, tree):
        ref = self.visit(tree.children[0])
        access = None
        if len(tree.children) > 1:
            access = self.visit(tree.children[1])
        else:
            return ref
        if isinstance(access, CallArgs):
            return FuncCall(ref, access)
        elif isinstance(access, Slice):
            return SliceExpr(ref, access)
        elif isinstance(access, FieldAccess):
            return FieldAccessExpr(ref, access)
        else:
            raise ValueError("Haven't handled other stuff yet")

    def sum_op(self, tree):
        return tree.children[0].value

    def term_op(self, tree):
        return tree.children[0].value

    def assign_op(self, tree):
        return tree.children[0].value

    def single_comp_op(self, tree):
        return tree.children[0].value

    def comp_op(self, tree):
        return self.visit(tree.children[0])

    def lor(self, tree):
        return tree.children[0].value

    def land(self, tree):
        return tree.children[0].value

    def lnot(self, tree):
        return tree.children[0].value

    def bor(self, tree):
        return tree.children[0].value

    def band(self, tree):
        return tree.children[0].value

    def bxor(self, tree):
        return tree.children[0].value

    def shift_op(self, tree):
        return tree.children[0].value

    def power_op(self, tree):
        return tree.children[0].value

    def factor_op(self, tree):
        return tree.children[0].value

    def binop(self, tree):
        if len(tree.children) == 1:
            return tree
        operator = self.visit(tree.children[1])
        lhs = self.visit(tree.children[0])
        rhs = self.visit(tree.children[2])
        return BinOp(lhs, operator, rhs)

    def unop(self, tree):
        if len(tree.children) == 1:
            return tree
        operator = self.visit(tree.children[0])
        rhs = self.visit(tree.children[1])
        return UnOp(operator, rhs)

    def term(self, tree):
        return self.binop(tree)

    def sum(self, tree):
        return self.binop(tree)

    def comparison(self, tree):
        return self.binop(tree)

    def bitwise_or(self, tree):
        return self.binop(tree)

    def bitwise_xor(self, tree):
        return self.binop(tree)

    def bitwise_and(self, tree):
        return self.binop(tree)

    def shift_expr(self, tree):
        return self.binop(tree)

    def power(self, tree):
        return self.binop(tree)

    def inversion(self, tree):
        return self.unop(tree)

    def factor(self, tree):
        return self.unop(tree)

    def more_disjunction(self, tree):
        operator = self.visit(tree.children[0])
        rhs = self.visit(tree.children[1])
        cont = self.visit_or_none(tree.children[2])
        if isinstance(cont, DisjRhs):
            return DisjRhs(operator, BinOp(rhs, cont.operator, cont.rhs))
        elif cont is None:
            return DisjRhs(operator, rhs)
        else:
            raise Exception("more_disjunction")

    def more_conjunction(self, tree):
        operator = self.visit(tree.children[0])
        rhs = self.visit(tree.children[1])
        cont = self.visit_or_none(tree.children[2])
        if isinstance(cont, ConjRhs):
            return ConjRhs(operator, BinOp(rhs, cont.operator, cont.rhs))
        elif cont is None:
            return ConjRhs(operator, rhs)
        else:
            raise Exception("more_conjunction")

    def disjunction(self, tree):
        lhs = self.visit(tree.children[0])
        operator = self.visit(tree.children[1])
        rhs = self.visit(tree.children[2])
        res = BinOp(lhs, operator, rhs)
        cont = self.visit_or_none(tree.children[3])
        if isinstance(cont, DisjRhs):
            res = BinOp(res, cont.operator, cont.rhs)
        elif cont is not None:
            raise Exception("disjunction")
        return res

    def conjunction(self, tree):
        lhs = self.visit(tree.children[0])
        operator = self.visit(tree.children[1])
        rhs = self.visit(tree.children[2])
        res = BinOp(lhs, operator, rhs)
        cont = self.visit_or_none(tree.children[3])
        if isinstance(cont, ConjRhs):
            res = BinOp(res, cont.operator, cont.rhs)
        elif cont is not None:
            raise Exception("conjunction")
        return res

    def if_expr(self, tree):
        guard = self.visit(tree.children[1])
        body = self.visit(tree.children[2])
        else_expr = self.visit(tree.children[3]) if tree.children[3] is not None else None
        return IfExpr(guard, body, else_expr)

    def elif_expr(self, tree):
        return self.if_expr(tree)

    def while_expr(self, tree):
        guard = self.visit(tree.children[1])
        body = self.visit(tree.children[2])
        else_expr = self.visit(tree.children[3]) if tree.children[3] is not None else None
        return WhileExpr(guard, body, else_expr)

    def elwhile_expr(self, tree):
        return self.while_expr(tree)

    def else_expr(self, tree):
        return self.visit(tree.children[1])

    def exit_direction(self, tree):
        return self.visit(tree.children[-1]).name

    def exit_expr(self, tree):
        return ExitExpr(tree.children[0].value,
                        self.visit(tree.children[1]) if tree.children[1] is not None else None,
                        self.visit(tree.children[2]) if tree.children[2] is not None else None)

    def param_eles(self, tree, pos_only=False, kw_only=False):
        return [self.visit(x, pos_only=pos_only, kw_only=kw_only) for x in tree.children]

    def param_ele(self, tree, pos_only=False, kw_only=False):
        return DeclArg(self.visit(tree.children[0]),
                       type_expr=self.visit_or_none(tree.children[1]),
                       default=self.visit_or_none(tree.children[2]),
                       pos_only=pos_only, kw_only=kw_only)

    def pos_or_kw_args(self, tree):
        args = self.visit(tree.children[0], pos_only=False, kw_only=False)
        va_args_kw_only = self.visit_or_default(tree.children[1], [])
        args.extend(va_args_kw_only)
        return args

    def fndecl(self, tree):
        name = self.visit_or_none(tree.children[1])
        args = DeclArgList(self.visit_or_none(tree.children[3]))
        type_expr = self.visit_or_none(tree.children[5])
        body = self.visit(tree.children[6])
        return FnDeclExpr(name, args, body, type_expr)

    def class_parent_ele(self, tree):
        return self.visit(tree.children[0]) # FIXME

    def class_parents(self, tree):
        return [self.visit(child) for child in tree.children]

    def classdecl(self, tree):
        class_type = tree.children[0].value
        name = self.visit_or_none(tree.children[1])
        parents = self.visit_or_none(tree.children[2])
        body = self.visit(tree.children[3])
        return ClassDecl(class_type, name, body, parents=parents)

    def assign(self, tree):
        target = self.visit(tree.children[0])
        type_expr = None
        if tree.children[1] is not None:
            type_expr = self.visit(tree.children[1])
        operator = self.visit(tree.children[2])
        value = self.visit(tree.children[3])
        return AssignOp(target, type_expr, operator, value)

    def for_in(self, tree):
        return self.visit(tree.children[0]), self.visit(tree.children[1])

    def for_expr(self, tree):
        induction_var, induction_expr = self.visit(tree.children[1])
        body = self.visit(tree.children[2])
        else_expr = self.visit_or_none(tree.children[3])
        return ForExpr(induction_var, induction_expr, body, else_expr)

    def group(self, tree):
        return Group([self.visit(x) for x in tree.children])

    def expr(self, tree):
        expr_to_eval = tree.children[0]
        if isinstance(expr_to_eval, Tree):
            return Expr(self.visit(expr_to_eval))
        elif isinstance(expr_to_eval, Token):
            return expr_to_eval.value
        return expr_to_eval

    def properties(self, tree):
        return set([x.value for x in tree.children])

    def expr_with_prop(self, tree):
        return Expr(self.visit(tree.children[1]).expr, properties=self.visit_or_none(tree.children[0]))

    def block(self, tree):
        return self.visit(tree.children[1]) if tree.children[1] is not None else ExprList([])

    def expr_list(self, tree):
        return ExprList([self.visit(child) for child in tree.children])

    def start(self, tree):
        return self.visit(tree.children[0])

class HognoseInterpreter:
    def __init__(self):
        grammar_text = None
        with open("grammar.lark") as f:
            grammar_text = f.read()
        self.parser = Lark(grammar_text, propagate_positions=True, ambiguity="explicit", lexer=HognoseLexer)
        self.scope = Scope(symbols={
            "print": BuiltinFnDef(print)
        })

    def parse(self, text, print_tree=False):
        raw_parse_res = self.parser.parse(text)
        try:
            parse_res = HognosePostParse().transform(raw_parse_res)
        except (VisitError, ParseError) as e:
            if print_tree:
                tree_print(raw_parse_res)
            raise e from None
        if print_tree:
            tree_print(parse_res)
        pre_ast = HognoseASTTransform(self.parser.rules).transform(parse_res)
        if print_tree:
            tree_print(pre_ast)
        tree_ast = HognoseASTGen().visit(pre_ast)
        if print_tree:
            tree_print(tree_ast)
        self.scope = tree_ast.eval(self.scope, return_scope=True)

    def parse_from_file(self, file_path, print_input=False, print_tree=False):
        input_text = None
        with open(file_path) as f:
            input_text = f.read()
        if print_input:
            for line_no, line in enumerate(input_text.splitlines()):
                print("{}: {}".format(line_no + 1, line))
        self.parse(input_text, print_tree=print_tree)

class HognoseRepl(Cmd):
    def __init__(self):
        super().__init__()
        self.__interp = HognoseInterpreter()
        self.intro = "\n".join([
        "Welcome to Hognose. This language is very under development.",
        "Type 'quit' or 'exit' to exit."
        ])
        self.__normal_prompt = "~~> "
        self.__oops_prompt = "~!> "
        self.prompt = self.__normal_prompt

    def default(self, line):
        try:
            self.__interp.parse(line + ";")
            self.prompt = self.__normal_prompt
        except (VisitError, ArgumentError, MissingSymbolError) as e:
            self.prompt = self.__oops_prompt
            print(e)

    def completenames(self, text, *args):
        ret_names = []
        if text in "exit":
            ret_names.append("exit")
        if text in "quit":
            ret_names.append("quit")
        if text in "help":
            ret_names.append("help")
        for symbol_name in self.__interp.scope.symbol_names():
            if text in symbol_name:
                ret_names.append(symbol_name)
        return ret_names

    def do_help(self, arg):
        if len(arg) == 0:
            self.stdout.write("help <cmd>: choose from '{}'\n".format(["exit", "quit", *self.__interp.scope.symbol_names()]))
            return
        if arg in ["exit", "help"]:
            self.stdout.write("Exit the REPL\n")
            return
        elif arg in ["help"]:
            self.stdout.write("This\n")
            return
        elif arg in self.__interp.scope.symbol_names():
            self.stdout.write("Hognose object: '{}'\n".format(self.__interp.scope.get(arg)))
            return
        self.stdout.write("Unknown symbol '{}'\n".format(arg))

    def emptyline(self):
        self.prompt = self.__normal_prompt
        pass

    def do_exit(self, line):
        return True

    def do_quit(self, line):
        return True

if len(sys.argv) > 1:
    HognoseInterpreter().parse_from_file(sys.argv[1], True, True)
else:
    try:
        HognoseRepl().cmdloop()
    except KeyboardInterrupt:
        print("")

#!/usr/bin/env python3

import sys
import ast
import re
from lark import Lark, Tree
from lark.lexer import Lexer, Token
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, LexError, ParseError, VisitError
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

class LexMatch:
    def __init__(self, terminal):
        self.pattern_str = terminal.pattern.to_regexp()
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

    def terminated_expr(self, tree):
        if all([x is None for x in tree.children]):
            return Discard
        return tree.children[1]

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
        ret_tree = tree.children[0]
        ret_tree.data = "call_args"
        return ret_tree

    def empty_expr(self, tree):
        return Discard

    def newlines(self, tree):
        return Discard

    def cond_body(self, tree):
        if tree.children[1] is not None:
            tree.children[1] = tree.children[1].children[0]
        return tree

    def elexpr(self, tree):
        control_expr = tree.children[0]
        cond_body = control_expr.children[2]
        del control_expr.children[2]
        control_expr.children.extend(cond_body.children)
        return control_expr

    def control_exprs(self, tree):
        control_expr = tree.children[0]
        cond_body = control_expr.children[2]
        del control_expr.children[2]
        control_expr.children.extend(cond_body.children)
        return control_expr

class Scope:
    def __init__(self, parent_scope=None, loop_scope=False, function_scope=False, symbols=None):
        self.parent_scope = parent_scope
        self.loop_scope = loop_scope
        self.function_scope = function_scope
        self.symbols = symbols if symbols is not None else {}
        self.break_called = False
        self.break_val = None
        self.defer_exprs = []

    def get(self, symbol):
        if symbol in self.symbols:
            return self.symbols[symbol]
        elif self.parent_scope is None:
            raise ValueError("No symbol '{}'".format(symbol))
        return self.parent_scope.get(symbol)

    def assign(self, symbol, value):
        if self.parent_scope is not None and self.parent_scope.has(symbol):
            return self.parent_scope.assign(symbol, value)
        self.symbols[symbol] = value
        return self.symbols[symbol]

    def has(self, symbol):
        return symbol in self.symbols or self.parent_scope.has(symbol) if self.parent_scope is not None else False

    def call_break(self, loop_break=False, function_break=False, loop_continue=False, break_val=None):
        if not (loop_continue is True and self.loop_scope is True):
            self.break_called = True
            self.break_val = None
        if self.parent_scope is not None:
            if (loop_break or loop_continue) and self.loop_scope is False:
                self.parent_scope.call_break(loop_break=loop_break, function_break=function_break, break_val=None)
            elif function_break and self.function_break is False:
                self.parent_scope.call_break(loop_break=loop_break, function_break=function_break, break_val=None)

    def add_defer(self, defer_expr):
        self.defer_exprs.insert(0, defer_expr)

    def has_defers(self):
        return len(self.defer_exprs) > 0

    def run_defers(self):
        last_res = None
        for defer_expr in self.defer_exprs:
            last_res = defer_expr.eval(self)
        return last_res

    def push_scope(self, loop_scope=False, function_scope=False, symbols=None):
        return Scope(parent_scope=self, loop_scope=loop_scope, function_scope=function_scope, symbols=symbols)

    def pop_scope(self):
        if self.parent_scope is None:
            raise ValueError("Cannot pop top scope")
        return self.parent_scope

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
        return str(self.value)

    def __repr__(self):
        return self.__str__()

class KwArg:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return "{}={}".format(self.name, str(self.value))

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
        pos_args = [x.eval(symbol_table) for x in self.args.pos_args]
        kw_args = {k: v.eval(symbol_table) for k, v in self.args.kw_args.items()}
        callee = self.callee.eval(symbol_table) if hasattr(self.callee, "eval") else self.callee
        return callee(*pos_args, **kw_args)

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
        rhs_val = self.rhs.eval(symbol_table)
        if self.operator == "+":
            return lhs_val + rhs_val
        elif self.operator == "-":
            return lhs_val - rhs_val
        elif self.operator == "*":
            return lhs_val * rhs_val
        elif self.operator == "/":
            return lhs_val / rhs_val
        elif self.operator == '>':
            return lhs_val > rhs_val
        elif self.operator == '>=':
            return lhs_val >= rhs_val
        elif self.operator == '<':
            return lhs_val < rhs_val
        elif self.operator == '<=':
            return lhs_val <= rhs_val
        elif self.operator == '==':
            return lhs_val == rhs_val
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

    def eval(self, symbol_table):
        target = self.target
        # TODO!!!
        if isinstance(target, NameRef):
            target = target.name
        elif not isinstance(target, str):
            target = target.eval(symbol_table)
        return symbol_table.assign(target, self.value.eval(symbol_table))

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
        symbol_table = symbol_table.pop_scope()
        if guard_res:
            return val
        elif self.else_expr is not None:
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

class ExprList:
    def __init__(self, exprs):
        self.exprs = list(exprs)

    def __str__(self):
        return "ExprList: {}".format("\n".join([str(x) for x in self.exprs]))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        last_res = None
        symbol_table = symbol_table.push_scope()
        for expr in self.exprs:
            last_res = expr.eval(symbol_table)
            if symbol_table.break_called:
                if symbol_table.break_val is not None:
                    last_res = symbol_table.break_val
                break
        symbol_table = symbol_table.pop_scope()
        return last_res

class HognoseASTGen(Interpreter):
    def __default__(self, tree):
        name = tree.data if hasattr(tree, "data") else tree.value
        raise ValueError("No handler for '{}'".format(name))

    def bin(self, tree):
        return LiteralInt(int(tree.chidren[0].value, 2))

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

    def name(self, tree):
        return NameRef(tree.children[0].value)

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
        if len(tree.children) > 1:
            raise ValueError("Not dealing with access yet")
        atom = self.visit(tree.children[0])
        if isinstance(atom, NameRef):
            return atom
        else:
            raise ValueError("Doesn't make sense yet")
        return atom

    def primary(self, tree):
        ref = self.visit(tree.children[0])
        access = None
        if len(tree.children) > 1:
            access = self.visit(tree.children[1])
        else:
            return ref
        if isinstance(access, CallArgs):
            return FuncCall(ref, access)
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

    def binop(self, tree):
        if len(tree.children) == 1:
            return tree
        operator = self.visit(tree.children[1])
        lhs = self.visit(tree.children[0])
        rhs = self.visit(tree.children[2])
        return BinOp(lhs, operator, rhs)

    def term(self, tree):
        return self.binop(tree)

    def sum(self, tree):
        return self.binop(tree)

    def comparison(self, tree):
        return self.binop(tree)

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

    def assign(self, tree):
        target = self.visit(tree.children[0])
        type_expr = None
        if tree.children[1] is not None:
            type_expr = self.visit(tree.children[1])
        operator = self.visit(tree.children[2])
        value = self.visit(tree.children[3])
        return AssignOp(target, type_expr, operator, value)

    def expr(self, tree):
        expr_to_eval = tree.children[0]
        if isinstance(expr_to_eval, Tree):
            return self.visit(expr_to_eval)
        elif isinstance(expr_to_eval, Token):
            return expr_to_eval.value
        return expr_to_eval

    def block(self, tree):
        return self.visit(tree.children[1])

    def expr_list(self, tree):
        return ExprList([self.visit(child) for child in tree.children])

    def start(self, tree):
        return self.visit(tree.children[0])

grammar_text = None

with open("grammar.lark") as f:
    grammar_text = f.read()

parser = Lark(grammar_text, propagate_positions=True, ambiguity="explicit", lexer=HognoseLexer)

input_text = None

with open(sys.argv[1]) as f:
    input_text = f.read()

for line_no, line in enumerate(input_text.splitlines()):
    print("{}: {}".format(line_no + 1, line))
raw_parse_res = parser.parse(input_text)
try:
    parse_res = HognosePostParse().transform(raw_parse_res)
except (VisitError, ParseError) as e:
    tree_print(raw_parse_res)
    raise e from None
tree_print(parse_res)

pre_ast = HognoseASTTransform(parser.rules).transform(parse_res)
tree_print(pre_ast)
tree_ast = HognoseASTGen().visit(pre_ast)
tree_print(tree_ast)
tree_ast.eval(Scope(symbols={"print": print}))

import os
from lark import Lark, Tree, Token
from lark.visitors import Interpreter, Transformer, v_args, Discard
from lark.exceptions import ParseError, VisitError, UnexpectedInput
from .lex import Lexer
from ..parser import BaseParser, ParserError, InternalParserError
from ..utils import tree_print
from ..ast import (
    LiteralTrue,
    LiteralFalse,
    LiteralNull,
    LiteralString,
    LiteralFloat,
    LiteralInt,
    LiteralList,
    LiteralDict,
    Group,
    RangeExpr,
    SliceExpr,
    Slice,
    FieldAccessExpr,
    FieldAccess,
    NameRef,
    PosArg,
    KwArg,
    CallArgs,
    FuncCall,
    BinOp,
    UnOp,
    ExitExpr,
    AssignOp,
    IfExpr,
    WhileExpr,
    DeclArg,
    VaDeclArg,
    KwVaDeclArg,
    DeclArgList,
    FnDeclExpr,
    OpDeclExpr,
    ForExpr,
    ClassParentDecl,
    ClassDecl,
    NamespaceDecl,
    DoubleStarExpr,
    StarExpr,
    UsingExpr,
    BlockExpr,
    ExprList,
    LabeledExpr,
    Expr
)

DEFAULT_GRAMMAR_NAME = "grammar.lark"
DEFAULT_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, DEFAULT_GRAMMAR_NAME)

@v_args(tree=True)
class ParseTreeGen(Transformer):
    def newlines(self, tree):
        return Discard

    def empty_expr(self, tree):
        return Discard

    def expr_list(self, tree):
        if len(tree.children) == 0:
            return Discard
        elif len(tree.children) == 1:
            if tree.children[0] is None:
                return Discard
            elif tree.children[0].data == tree.data:
                return tree.children[0]
        return tree

    def _ambig(self, tree):
        if isinstance(tree, list) and len(tree) == 1:
            return tree[0]
        elif isinstance(tree, Tree):
            return tree
        ambig_names = set([x.data.value for x in tree])
        if len(ambig_names) > 1:
            raise InternalParserError("Cannot resolve ambiguity for more than one rule type: {}".format(list(ambig_names)))
        ambig_name = list(ambig_names)[0]
        ambig_handler = "_ambig_{}".format(ambig_name)
        if not hasattr(self, ambig_handler):
            raise InternalParserError("No ambiguity handler registered for {}\n\n{}".format(ambig_name, "\n\n".join([x.pretty() for x in tree])))
        return getattr(self, ambig_handler)(tree)

class ConjRhs:
    def __init__(self, operator, rhs):
        self.operator = operator
        self.rhs = rhs

class DisjRhs:
    def __init__(self, operator, rhs):
        self.operator = operator
        self.rhs = rhs

class ASTGen(Interpreter):
    def visit(self, tree):
        if isinstance(tree, Tree):
            if not hasattr(self, tree.data):
                raise InternalParserError("No handler for '{}'".format(tree.data))
            return getattr(self, tree.data)(tree)
        elif isinstance(tree, Token):
            return tree.value
        else:
            raise InternalParserError("Unhandled tree type: '{}'".format(type(tree).__name__))

    def visit_or_default(self, tree, default):
        if tree is None:
            return default
        return self.visit(tree)

    def visit_or_none(self, tree):
        return self.visit_or_default(tree, None)

    def __default__(self, tree):
        raise InternalParserError("No handler for '{}'".format(tree.data))

    def start(self, tree):
        return self.visit(tree.children[0])

    def expr_list(self, tree):
        expr_list = []
        expr = self.visit_or_none(tree.children[0])
        if expr is not None:
            expr_list.append(expr)
        if len(tree.children) > 1:
            more_expr_list = self.visit_or_none(tree.children[1])
            if more_expr_list is not None:
                expr_list.extend(more_expr_list.exprs)
        return ExprList(expr_list)

    def terminated_or_empty_expr(self, tree):
        return self.visit(tree.children[0])

    def terminated_expr(self, tree):
        return self.visit(tree.children[0])

    def empty_expr(self, tree):
        return None

    def expr_with_prop(self, tree):
        expr = self.visit(tree.children[1])
        properties = self.visit_or_none(tree.children[0])
        return Expr(expr.expr, properties=properties, type_expr=expr.type_expr)

    def properties(self, tree):
        property_list = None
        property = self.visit_or_none(tree.children[0])
        if property is not None:
            property_list = [property]
            more_properties = self.visit_or_none(tree.children[1])
            if more_properties is not None:
                property_list.extend(more_properties)
        return property_list

    def more_properties(self, tree):
        return self.properties(tree)

    def property(self, tree):
        return self.visit(tree.children[0])

    def expr(self, tree):
        expr = self.visit(tree.children[0])
        type_expr = self.visit_or_none(tree.children[1])
        if isinstance(expr, list):
            raise InternalParserError("list is an error:", tree.children[0].data)
        return Expr(expr, type_expr=type_expr)

    def using_expr(self, tree):
        target = self.visit(tree.children[1])
        as_expr = self.visit_or_none(tree.children[2])
        return UsingExpr(target, as_expr=as_expr)

    def using_as(self, tree):
        return self.visit(tree.children[1])

    def exit_expr(self, tree):
        exit_type = self.visit(tree.children[0])
        exit_val = self.visit_or_none(tree.children[1])
        exit_dir = self.visit_or_none(tree.children[2])
        return ExitExpr(exit_type, exit_val=exit_val, exit_dir=exit_dir)

    def exit_type(self, tree):
        return self.visit(tree.children[0])

    def exit_direction(self, tree):
        return self.visit(tree.children[1])

    def type_expr(self, tree):
        return self.visit(tree.children[1])

    def assign(self, tree):
        target = self.visit(tree.children[0])
        operator = self.visit(tree.children[2])
        value = self.visit(tree.children[3])
        type_expr = self.visit_or_none(tree.children[1])
        return AssignOp(target, operator, value, type_expr=type_expr)

    def group(self, tree):
        expr_list = [self.visit(tree.children[0])]
        more_expr = self.visit_or_none(tree.children[1])
        if more_expr is not None:
            expr_list.extend(more_expr)
        return Group(expr_list)

    def more_group(self, tree):
        if len(tree.children) == 1:
            return None
        expr_list = [self.visit(tree.children[1])]
        more_expr = self.visit_or_none(tree.children[2])
        if more_expr is not None:
            expr_list.extend(more_expr)
        return expr_list

    def single_star_expr(self, tree):
        return StarExpr(self.visit(tree.children[1]))

    def double_star_expr(self, tree):
        return DoubleStarExpr(self.visit(tree.children[1]))

    def fndecl(self, tree):
        name = self.visit_or_none(tree.children[1])
        args = DeclArgList(self.visit_or_none(tree.children[3]))
        type_expr = self.visit_or_none(tree.children[5])
        body = self.visit(tree.children[6])
        return FnDeclExpr(name, args, body, type_expr)

    def fnparams(self, tree):
        return self.visit(tree.children[0])

    def pos_only_params(self, tree):
        params = [DeclArg(x.name, type_expr=x.type_expr, default=x.default, pos_only=True) for x in self.visit(tree.children[0])]
        pos_or_kw_args = self.visit_or_none(tree.children[3])
        if pos_or_kw_args is not None:
            params.extend(pos_or_kw_args)
        return params

    def opt_pos_or_kw_args(self, tree):
        return self.visit(tree.children[1])

    def pos_or_kw_args(self, tree):
        params = self.visit(tree.children[0])
        va_and_kw_only_args = self.visit_or_none(tree.children[1])
        if va_and_kw_only_args is not None:
            params.extend(va_and_kw_only_args)
        return params

    def opt_va_args_kw_only(self, tree):
        return self.visit(tree.children[1])

    def va_args_kw_only(self, tree):
        params = None
        va_args = self.visit(self.children[0])
        if va_args is not None:
            params = [va_args]
        opt_params = [DeclArg(x.name, type_expr=x.type_expr, default=x.default, kw_only=True) for x in self.visit(tree.children[1])] if tree.children[1] is not None else None
        if opt_params is not None:
            params.extend(opt_params)
        kw_va_args = self.visit_or_none(self.children[2])
        if kw_va_args is not None:
            params.extend(kw_va_args)
        return params

    def star_param_ele_or_star(self, tree):
        param = self.visit(tree.children[0])
        if isinstance(param, VaDeclArg):
            return param
        return None

    def opt_kw_va_args(self, tree):
        return self.visit(tree.children[1])

    def kw_va_args(self, tree):
        return self.visit(tree.children[0])

    def opt_param_eles(self, tree):
        return self.visit(tree.children[1])

    def param_eles(self, tree):
        params = [self.visit(tree.children[0])]
        more_params = self.visit_or_none(tree.children[1])
        if more_params is not None:
            params.extend(more_params)
        return params

    def param_ele(self, tree):
        name = self.visit(tree.children[0])
        type_expr = self.visit_or_none(tree.children[1])
        default = self.visit_or_none(tree.children[2])
        return DeclArg(name, type_expr=type_expr, default=default)

    def more_param_eles(self, tree):
        params = [self.visit(tree.children[1])]
        more_params = self.visit_or_none(tree.children[2])
        if more_params is not None:
            params.extend(more_params)
        return params

    def star_param_ele(self, tree):
        name = self.visit(tree.children[0])
        type_expr = self.visit_or_none(tree.children[1])
        return VaDeclArg(name, type_expr=type_expr)

    def double_star_param_ele(self, tree):
        name = self.visit(tree.children[0])
        type_expr = self.visit_or_none(tree.children[1])
        return KwVaDeclArg(name, type_expr=type_expr)

    def param_default(self, tree):
        return self.visit(tree.children[1])

    def operatordecl(self, tree):
        operator = self.visit(tree.children[1])
        args = DeclArgList(self.visit_or_none(tree.children[3]))
        type_expr = self.visit_or_none(tree.children[5])
        body = self.visit(tree.children[6])
        return OpDeclExpr(operator, args, body, type_expr)

    def classdecl(self, tree):
        class_type = self.visit(tree.children[0])
        name = self.visit_or_none(tree.children[1])
        parents = self.visit_or_none(tree.children[2])
        body = self.visit(tree.children[3])
        return ClassDecl(class_type, name, body, parents=parents)

    def class_type(self, tree):
        return self.visit(tree.children[0])

    def class_parents(self, tree):
        parent_eles = [self.visit(tree.children[1])]
        more_eles = self.visit_or_none(tree.children[2])
        if more_eles is not None:
            parent_eles.extend(more_eles)
        return parent_eles

    def more_class_parents(self, tree):
        return self.class_parents(tree)

    def class_parent_ele(self, tree):
        return self.visit(tree.children[0])

    def parent_name(self, tree):
        name = self.visit(tree.children[0])
        assignment = self.visit_or_none(tree.children[1])
        return ClassParentDecl(name, assignment=assignment)

    def class_parent_assign(self, tree):
        return self.visit(tree.children[1])

    def namespace_whole_file(self, tree):
        name = self.visit(tree.children[1])
        return NamespaceDecl(name=name)

    def namespace_local(self, tree):
        name = self.visit_or_none(tree.children[1])
        body = self.visit(tree.children[2])
        return NamespaceDecl(name=name, body=body)

    def label(self, tree):
        return self.visit(tree.children[0])

    def labeled_block(self, tree):
        label = self.visit(tree.children[0])
        block = self.visit(tree.children[1])
        return LabeledExpr(label, block)

    def block(self, tree):
        exprs = self.visit_or_default(tree.children[1], ExprList([]))
        return BlockExpr(exprs)

    def labeled_control_expr(self, tree):
        label = self.visit(tree.children[0])
        expr = self.visit(tree.children[1])
        return LabeledExpr(label, expr)

    def cond_body(self, tree):
        return self.visit(tree.children[0]), self.visit_or_none(tree.children[1])

    def if_expr(self, tree):
        guard = self.visit(tree.children[1])
        body, elexpr = self.visit(tree.children[2])
        return IfExpr(guard, body, elexpr=elexpr)

    def elif_expr(self, tree):
        return self.if_expr(tree)

    def for_in(self, tree):
        return self.visit(tree.children[0]), self.visit(tree.children[2])

    def for_expr(self, tree):
        induction_var, induction_expr = self.visit(tree.children[1])
        body, elexpr = self.visit(tree.children[2])
        return ForExpr(induction_var, induction_expr, body, elexpr=elexpr)

    def elfor_expr(self, tree):
        return self.for_expr(tree)

    def while_expr(self, tree):
        guard = self.visit(tree.children[1])
        body, elexpr = self.visit(tree.children[2])
        return WhileExpr(guard, body, elexpr=elexpr)

    def elexpr_or_else(self, tree):
        return self.visit(tree.children[0])

    def elexpr_or_else_real(self, tree):
        return self.visit(tree.children[0])

    def elexpr(self, tree):
        return self.visit(tree.children[0])

    def else_expr(self, tree):
        return self.visit(tree.children[1])

    def arith(self, tree):
        return self.visit(self.children[0])

    def range(self, tree):
        start, start_closed = self.visit(tree.children[0])
        step = self.visit_or_none(tree.children[1])
        end, end_closed = self.visit(tree.children[3])
        return RangeExpr(start=start, start_closed=start_closed,
                         end=end, end_closed=end_closed, step=step)

    def range_step(self, tree):
        return self.visit(tree.children[1])

    def range_start(self, tree):
        start = self.visit(tree.children[0])
        start_closed = False if tree.children[1] is None else True
        return start, start_closed

    def range_start_inner(self, tree):
        expr = self.visit(tree.children[0])
        if expr == "<":
            return None
        return expr

    def range_end(self, tree):
        end = self.visit(tree.children[1])
        end_closed = False if tree.children[0] is None else True
        return end, end_closed

    def range_end_inner(self, tree):
        expr = self.visit(tree.children[0])
        if expr == ">":
            return None
        return expr

    def disjunction(self, tree):
        lhs = self.visit(tree.children[0])
        operator = self.visit(tree.children[1])
        rhs = self.visit(tree.children[2])
        res = BinOp(lhs, operator, rhs)
        cont = self.visit_or_none(tree.children[3])
        if isinstance(cont, DisjRhs):
            res = BinOp(res, cont.operator, cont.rhs)
        elif cont is not None:
            raise InternalParserError("Unexpected disjunction continuation type: '{}'".format(type(cont).__name__))
        return res

    def more_disjunction(self, tree):
        operator = self.visit(tree.children[0])
        rhs = self.visit(tree.children[1])
        cont = self.visit_or_none(tree.children[2])
        if isinstance(cont, DisjRhs):
            return DisjRhs(operator, BinOp(rhs, cont.operator, cont.rhs))
        elif cont is None:
            return DisjRhs(operator, rhs)
        raise InternalParserError("Unexpected disjunction continuation type: '{}'".format(type(cont).__name__))

    def conjunction(self, tree):
        lhs = self.visit(tree.children[0])
        operator = self.visit(tree.children[1])
        rhs = self.visit(tree.children[2])
        res = BinOp(lhs, operator, rhs)
        cont = self.visit_or_none(tree.children[3])
        if isinstance(cont, ConjRhs):
            res = BinOp(res, cont.operator, cont.rhs)
        elif cont is not None:
            raise InternalParserError("Unexpected conjunction continuation type: '{}'".format(type(cont).__name__))
        return res

    def more_conjunction(self, tree):
        operator = self.visit(tree.children[0])
        rhs = self.visit(tree.children[1])
        cont = self.visit_or_none(tree.children[2])
        if isinstance(cont, ConjRhs):
            return ConjRhs(operator, BinOp(rhs, cont.operator, cont.rhs))
        elif cont is None:
            return ConjRhs(operator, rhs)
        raise InternalParserError("Unexpected conjunction continuation type: '{}'".format(type(cont).__name__))

    def binop(self, tree):
        operator = self.visit(tree.children[1])
        lhs = self.visit(tree.children[0])
        rhs = self.visit(tree.children[2])
        return BinOp(lhs, operator, rhs)

    def unop(self, tree):
        operator = self.visit(tree.children[0])
        rhs = self.visit(tree.children[1])
        return UnOp(operator, rhs)

    def inversion(self, tree):
        return self.unop(tree)

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

    def sum(self, tree):
        return self.binop(tree)

    def term(self, tree):
        return self.binop(tree)

    def factor(self, tree):
        return self.unop(tree)

    def power(self, tree):
        return self.binop(tree)

    def primary(self, tree):
        ref = self.visit(tree.children[0])
        if len(tree.children) == 1:
            return ref
        access = self.visit_or_none(tree.children[1])
        if access is None:
            return ref
        if isinstance(access, CallArgs):
            return FuncCall(ref, access)
        elif isinstance(access, Slice):
            return SliceExpr(ref, access)
        elif isinstance(access, FieldAccess):
            return FieldAccessExpr(ref, access)
        raise InternalParserError("Unexpected primary access type: '{}'".format(type(access).__name__))

    def field_access(self, tree):
        return FieldAccess(self.visit(tree.children[1]))

    def slice(self, tree):
        return Slice(self.visit(tree.children[0]))

    def call_args(self, tree):
        args = self.visit_or_none(tree.children[1])
        return CallArgs(args)

    def call_args_body(self, tree):
        args = [self.visit(tree.children[0])]
        more_args = self.visit_or_none(tree.children[1])
        if more_args is not None:
            args.extend(more_args)
        return args

    def more_call_args(self, tree):
        args = [self.visit(tree.children[1])]
        more_args = self.visit_or_none(tree.children[2])
        if more_args is not None:
            args.extend(more_args)
        return args

    def kw_arg(self, tree):
        return KwArg(self.visit(tree.children[0]).name, self.visit(tree.children[2]))

    def pos_arg(self, tree):
        return PosArg(self.visit(tree.children[0]))

    def slice(self, tree):
        return Slice(self.visit(tree.children[1]))

    def parens(self, tree):
        return self.visit(tree.children[1])

    def list(self, tree):
        list_elements = self.visit_or_default(tree.children[1], list())
        return LiteralList(list_elements)

    def list_body(self, tree):
        return self.visit(tree.children[0])

    def list_elements(self, tree):
        elements = [self.visit(tree.children[0])]
        more_elements = self.visit_or_none(tree.children[1])
        if more_elements is not None and more_elements != ",":
            elements.extend(more_elements)
        return elements

    def list_element(self, tree):
        return self.visit(tree.children[0])

    def more_list_elements(self, tree):
        if len(tree.children) == 1:
            return None
        elements = [self.visit(tree.children[1])]
        more_elements = self.visit_or_none(tree.children[2])
        if more_elements is not None and more_elements != ",":
            elements.extend(more_elements)
        return elements

    def dict(self, tree):
        dict_elements = self.visit(tree.children[1])
        if dict_elements is None or dict_elements == ":":
            dict_elements = dict()
        else:
            dict_elements = {k: v for k, v in dict_elements}
        return LiteralDict(dict_elements)

    def dict_body(self, tree):
        dict_elements = self.visit(tree.children[0])
        if dict_elements == ":":
            return None
        return dict_elements

    def dict_elements(self, tree):
        dict_elements = [self.visit(tree.children[0])]
        more_elements = self.visit_or_none(tree.children[1])
        if more_elements is not None and more_elements != ",":
            dict_elements.extend(more_elements)
        return dict_elements

    def dict_element(self, tree):
        return (self.visit(tree.children[0]), self.visit(tree.children[2]))

    def more_dict_elements(self, tree):
        if len(tree.children) == 1:
            return None
        dict_elements = [self.visit(tree.children[1])]
        more_elements = self.visit_or_none(tree.children[2])
        if more_elements is not None and more_elements != ",":
            dict_elements.extend(more_elements)
        return dict_elements

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

    def d_ml_string(self, tree): # FIXME pretty sure this is wrong
        new_string = tree.children[0].value[3:-3]
        return LiteralString(str(new_string))

    def s_ml_string(self, tree):
        new_string = tree.children[0].value[3:-3]
        return LiteralString(str(new_string))

    def d_sl_string(self, tree):
        new_string = tree.children[0].value[1:-1]
        return LiteralString(str(new_string))

    def s_sl_string(self, tree):
        new_string = tree.children[0].value[1:-1]
        return LiteralString(str(new_string))

    def string(self, tree):
        return self.visit(tree.children[0])

    def comp_op(self, tree):
        return self.visit(tree.children[0])

    def single_comp_op(self, tree):
        return self.visit(tree.children[0])

    def multi_comp_op(self, tree):
        return self.visit(tree.children[0])

    def not_in(self, tree):
        return "{} {}".format(self.visit(tree.children[0]), self.visit(tree.children[1]))

    def in_op(self, tree):
        return self.visit(tree.children[0])

    def is_not(self, tree):
        return "{} {}".format(self.visit(tree.children[0]), self.visit(tree.children[1]))

    def is_op(self, tree):
        return self.visit(tree.children[0])

    def has(self, tree):
        return self.visit(tree.children[0])

    def of(self, tree):
        return self.visit(tree.children[0])

    def power_op(self, tree):
        return self.visit(tree.children[0])

    def factor_op(self, tree):
        return self.visit(tree.children[0])

    def term_op(self, tree):
        return self.visit(tree.children[0])

    def sum_op(self, tree):
        return self.visit(tree.children[0])

    def shift_op(self, tree):
        return self.visit(tree.children[0])

    def band(self, tree):
        return self.visit(tree.children[0])

    def bxor(self, tree):
        return self.visit(tree.children[0])

    def bor(self, tree):
        return self.visit(tree.children[0])

    def land(self, tree):
        return self.visit(tree.children[0])

    def lor(self, tree):
        return self.visit(tree.children[0])

    def lnot(self, tree):
        return self.visit(tree.children[0])

    def assign_op(self, tree):
        return self.visit(tree.children[0])

    def dot_op(self, tree):
        return self.visit(tree.children[0])

    def array_op(self, tree):
        return "{}{}".format(self.visit(tree.children[0]), self.visit(tree.children[1]))

    def op(self, tree):
        return self.visit(tree.children[0])

    def true(self, tree):
        return LiteralTrue()

    def false(self, tree):
        return LiteralFalse()

    def null(self, tree):
        return LiteralNull()

    def name(self, tree):
        return NameRef(self.visit(tree.children[0]))

    def super_ref(self, tree):
        return NameRef(self.visit(tree.children[0]))

    def self_ref(self, tree):
        return NameRef(self.visit(tree.children[0]))

    def cls_ref(self, tree):
        return NameRef(self.visit(tree.children[0]))

class LarkParser(BaseParser):
    def __init__(self, grammar_file=DEFAULT_GRAMMAR_PATH):
        super().__init__(grammar_file)
        self.parser = Lark(self.grammar_text, propagate_positions=True, ambiguity="explicit", lexer=Lexer)

    def parse(self, text, print_tree=False):
        try:
            raw_parse_res = self.parser.parse(text)
        except UnexpectedInput as e:
            raise ParserError from e
        try:
            parse_res = ParseTreeGen().transform(raw_parse_res)
        except (VisitError, ParseError) as e:
            if print_tree:
                tree_print(raw_parse_res)
            raise ParserError from e
        if print_tree:
            tree_print(parse_res)
        tree_ast = ASTGen().visit(parse_res)
        if print_tree:
            tree_print(tree_ast)
        return tree_ast

import os
import inspect
from ..parser import BaseParser, ParserError, InternalParserError
from parsimonious.grammar import Grammar
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

DEFAULT_GRAMMAR_NAME = "grammar.ppeg"
DEFAULT_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, DEFAULT_GRAMMAR_NAME)

class BinRhs:
    def __init__(self, operator, rhs):
        self.operator = operator
        self.rhs = rhs

class ASTGen:
    def start(self, node):
        return self.visit(node.children[0])

    def expr_list(self, node):
        statements = self.visit(node.children[1])
        if statements is None:
            return None
        elif not isinstance(statements, list):
            statements = [statements]
        return ExprList(statements)

    def statements(self, node):
        statements = [self.visit(node.children[0])]
        more_statements = self.visit_or_none(node.children[1])
        if more_statements is not None:
            statements.extend(more_statements)
        return [x for x in statements if x is not None]

    def more_statements(self, node):
        return self.statements(node)

    def statement(self, node):
        return self.visit(node.children[0])

    def terminated_expr(self, node):
        properties = self.visit_or_none(node.children[1])
        expr = self.visit(node.children[2])
        return Expr(expr.expr, type_expr=expr.type_expr, properties=properties)

    def empty_expr(self, node):
        return None

    def properties(self, node):
        properties = [self.visit(node.children[0])]
        more_properties = self.visit_or_none(node.children[1])
        if more_properties is not None:
            properties.extend(more_properties)
        return [x for x in properties if x is not None]

    def property(self, node):
        return self.visit(node.children[0])

    def property_name(self, node):
        return node.children[0].text

    def expr(self, node):
        expr = self.visit(node.children[0])
        type_expr = self.visit_or_none(node.children[1])
        return Expr(expr, type_expr=type_expr)

    def group(self, node):
        expr = self.visit(node.children[0])
        more_exprs = self.visit_or_none(node.children[1])
        if more_exprs is not None:
            exprs = [expr]
            exprs.extend(more_exprs)
            return Group(exprs)
        return expr

    def more_group(self, node):
        exprs = [self.visit(node.children[3])]
        more_exprs = self.visit_or_none(node.children[4])
        if more_exprs is not None:
            exprs.extend(more_exprs)
        return exprs

    def single_expr(self, node):
        return self.visit(node.children[0])

    def using_expr(self, node):
        target = self.visit(node.children[2])
        as_expr = self.visit_or_none(node.children[3])
        return UsingExpr(target, as_expr=as_expr)

    def exit_expr(self, node):
        exit_type = self.visit(node.children[0])
        exit_val = self.visit_or_none(node.children[1])
        exit_dir = self.visit_or_none(node.children[2])
        return ExitExpr(exit_type, exit_val=exit_val, exit_dir=exit_dir)

    def exit_inner_expr(self, node):
        return self.visit(node.children[1])

    def exit_direction(self, node):
        return self.visit(node.children[3])

    def exit_type(self, node):
        return node.children[0].text

    def type_expr(self, node):
        return self.visit(node.children[3])

    def assign(self, node):
        target = self.visit(node.children[0])
        type_expr = self.visit_or_none(node.children[1])
        assign_op = self.visit(node.children[3])
        value = self.visit(node.children[5])
        return AssignOp(target, assign_op, value, type_expr=type_expr)

    def starred_expr(self, node):
        return self.visit(node.children[0])

    def single_star_expr(self, node):
        expr = self.visit(node.children[2])
        return StarExpr(expr)

    def double_star_expr(self, node):
        expr = self.visit(node.children[2])
        return DoubleStarExpr(expr)

    def structure_decl(self, node):
        return self.visit(node.children[0])

    def fndecl(self, node):
        name = self.visit_or_none(node.children[1])
        params = self.visit_or_none(node.children[5])
        if isinstance(params, list):
            params = [x for x in params if x is not None]
        args = DeclArgList(params)
        type_expr = self.visit_or_none(node.children[7])
        body = self.visit(node.children[9])
        return FnDeclExpr(name, args, body, type_expr=type_expr)

    def fn_name(self, node):
        return self.visit(node.children[1])

    def fnparams(self, node):
        return self.visit(node.children[0])

    def pos_only_params(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        param_eles = [DeclArg(x, pos_only=True) for x in self.visit(node.children[0])]
        more_param_eles = self.visit_or_none(node.children[5])
        if more_param_eles is not None:
            param_eles.extend(more_param_eles)
        return param_eles

    def opt_pos_or_kw_args(self, node):
        return self.visit(node.children[2])

    def pos_or_kw_args(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        param_eles = self.visit(node.children[0])
        more_param_eles = self.visit_or_none(node.children[1])
        if more_param_eles is not None:
            param_eles.extend(more_param_eles)
        return param_eles

    def opt_va_args_kw_only(self, node):
        return self.visit(node.children[2])

    def va_args_kw_only(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        param_eles = []
        va_args = self.visit(node.children[0])
        if va_args is not None:
            param_eles.append(va_args)
        opt_param_eles = self.visit_or_none(node.children[1])
        if opt_param_eles is not None:
            param_eles.extend(opt_param_eles)
        opt_kw_va_args = self.visit_or_none(node.children[2])
        if opt_kw_va_args is not None:
            param_eles.extend(opt_kw_va_args)
        return param_eles

    def star_param_ele_or_star(self, node):
        if node.children[0].expr.name == "STAR":
            return None
        return self.visit(node.children[0])

    def opt_kw_va_args(self, node):
        return self.visit(node.children[2])

    def kw_va_args(self, node):
        return self.visit(node.children[0])

    def opt_param_eles(self, node):
        return self.visit(node.children[2])

    def param_eles(self, node):
        param_eles = [self.visit(node.children[0])]
        more_param_eles = self.visit_or_none(node.children[1])
        if more_param_eles is not None:
            param_eles.extend(more_param_eles)
        return param_eles

    def param_ele(self, node):
        name = self.visit(node.children[0])
        type_expr = self.visit_or_none(node.children[1])
        default = self.visit_or_none(node.children[2])
        return DeclArg(name, default=default, type_expr=type_expr)

    def more_param_eles(self, node):
        param_eles = [self.visit(node.children[2])]
        more_param_eles = self.visit_or_none(node.children[3])
        if more_param_eles is not None:
            param_eles.extend(more_param_eles)
        return param_eles

    def star_param_ele(self, node):
        name = self.visit(node.children[1])
        type_expr = self.visit_or_none(node.children[2])
        return VaDeclArg(name, type_expr=type_expr)

    def double_star_param_ele(self, node):
        name = self.visit(node.children[1])
        type_expr = self.visit_or_none(node.children[2])
        return KwVaDeclArg(name, type_expr=type_expr)

    def param_default(self, node):
        expr = self.visit(node.children[3])
        return expr

    def operatordecl(self, node):
        operator = self.visit_or_none(node.children[2])
        params = self.visit_or_none(node.children[6])
        if isinstance(params, list):
            params = [x for x in params if x is not None]
        args = DeclArgList(params)
        type_expr = self.visit_or_none(node.children[8])
        body = self.visit(node.children[10])
        return OpDeclExpr(operator, args, body, type_expr=type_expr)


    def classdecl(self, node):
        class_type = self.visit(node.children[0])
        name = None
        if len(node.children) > 2 and node.children[2].expr_name.lower() == "name":
            name = self.visit(node.children[2])
        parents = None
        if node.children[1].expr_name == "class_parents":
            parents = self.visit(node.children[1])
        elif len(node.children) > 3 and node.children[3].expr_name == "class_parents":
            parents = self.visit(node.children[3])
        body = self.visit(node.children[-1])
        return ClassDecl(class_type, name=name, body=body, parents=parents)

    def class_type(self, node):
        return node.children[0].text

    def class_parents(self, node):
        parents = [self.visit(node.children[3])]
        more_parents = self.visit_or_none(node.children[4])
        if more_parents is not None:
            parents.extend(more_parents)
        return parents

    def more_class_parents(self, node):
        return self.class_parents(node)

    def parent_name(self, node):
        name = self.visit(node.children[0])
        assignment = self.visit_or_none(node.children[1])
        return ClassParentDecl(name, assignment=assignment)

    def class_parent_assign(self, node):
        return self.visit(node.children[3])

    def namespacedecl(self, node):
        return self.visit(node.children[0])

    def namespace_whole_file(self, node):
        name = self.visit(node.children[2])
        return NamespaceDecl(name=name)

    def namespace_local(self, node):
        name = self.visit_or_none(node.children[1])
        body = self.visit(node.children[3])
        return NamespaceDecl(name=name, body=body)

    def namespace_name(self, node):
        return self.visit(node.children[1])

    def label(self, node):
        return self.visit(node.children[0])

    def labeled_expr(self, node):
        label = self.visit_or_none(node.children[0])
        expr = self.visit(node.children[1])
        if label is not None:
            return LabeledExpr(label, expr)
        return expr

    def labeled_block(self, node):
        return self.labeled_expr(node)

    def block(self, node):
        exprs = self.visit(node.children[2])
        return ExprList(exprs)

    def control(self, node):
        return self.labeled_expr(node)

    def control_expr(self, node):
        return self.visit(node.children[0])

    def cond_body(self, node):
        expr = self.visit(node.children[1])
        elexpr = None
        if len(node.children) > 2:
            elexpr = self.visit(node.children[2])
        return expr, elexpr

    def if_expr(self, node):
        guard = self.visit(node.children[2])
        body, elexpr = self.visit(node.children[3])
        return IfExpr(guard, body, elexpr=elexpr)

    def elif_expr(self, node):
        return self.if_expr(node)

    def for_in(self, node):
        induction_vars = self.visit(node.children[1])
        induction_expr = self.visit(node.children[5])
        return induction_vars, induction_expr

    def for_expr(self, node):
        induction_vars, induction_expr = self.visit(node.children[1])
        body, elexpr = self.visit(node.children[2])
        return ForExpr(induction_vars, induction_expr, body, elexpr=elexpr)

    def elfor_expr(self, node):
        return self.for_expr(node)

    def while_expr(self, node):
        guard = self.visit(node.children[2])
        body, elexpr = self.visit(node.children[3])
        return WhileExpr(guard, body, elexpr=elexpr)

    def elwhile_expr(self, node):
        return self.while_expr(node)

    def elexpr_or_else_expr(self, node):
        return self.visit(node.children[1])

    def elexpr(self, node):
        return self.visit(node.children[0])

    def else_expr(self, node):
        return self.visit(node.children[2])

    def arith(self, node):
        return self.visit(node.children[0])

    def range(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        start, start_closed = self.visit(node.children[0])
        step = None
        if len(node.children) > 8:
            step = self.visit(node.children[4])
            end, end_closed = self.visit(node.children[8])
        else:
            end, end_closed = self.visit(node.children[4])
        return RangeExpr(start=start, start_closed=start_closed,
                         end=end, end_closed=end_closed, step=step)

    def range_step(self, node):
        return self.visit(node.children[3])

    def range_start(self, node):
        start = self.visit(node.children[0])
        if start == "<":
            start = None
        start_closed = False
        if len(node.children) > 1:
            start_closed = True
        return start, start_closed

    def range_end(self, node):
        end = None
        end_closed = False
        if len(node.children) == 1 and node.children[0].expr.name == "disjunction":
            end = self.visit(node.children[0])
        elif len(node.children) == 2:
            end_closed = True
        elif len(node.children) == 3:
            end_closed = True
            end = self.visit(node.children[2])
        return end, end_closed

    def disjunction(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        lhs = self.visit(node.children[0])
        cont = self.visit(node.children[1])
        return BinOp(lhs, cont.operator, cont.rhs)

    def more_disjunction(self, node):
        operator = self.visit(node.children[1])
        rhs = self.visit(node.children[3])
        cont = self.visit_or_none(node.children[4])
        if isinstance(cont, BinRhs):
            return BinRhs(operator, BinOp(rhs, cont.operator, cont.rhs))
        return BinRhs(operator, rhs)

    def conjunction(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        lhs = self.visit(node.children[0])
        cont = self.visit(node.children[1])
        return BinOp(lhs, cont.operator, cont.rhs)

    def more_conjunction(self, node):
        operator = self.visit(node.children[1])
        rhs = self.visit(node.children[3])
        cont = self.visit_or_none(node.children[4])
        if isinstance(cont, BinRhs):
            return BinRhs(operator, BinOp(rhs, cont.operator, cont.rhs))
        return BinRhs(operator, rhs)

    def unop(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        operator = self.visit(node.children[0])
        rhs = self.visit(node.children[2])
        return UnOp(operator, rhs)

    def binop(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        lhs = self.visit(node.children[0])
        cont = self.visit_or_none(node.children[1])
        if cont is None:
            return lhs
        return BinOp(lhs, cont.operator, cont.rhs)

    def binop_cont(self, node):
        operator = self.visit(node.children[1])
        rhs = self.visit(node.children[3])
        cont = self.visit_or_none(node.children[4])
        if cont is None:
            return BinRhs(operator, rhs)
        return BinRhs(operator, BinOp(rhs, cont.operator, cont.rhs))

    def inversion(self, node):
        return self.unop(node)

    def comparison(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        lhs = self.visit(node.children[0])
        operator = self.visit(node.children[2])
        rhs = self.visit(node.children[4])
        return BinOp(lhs, operator, rhs)

    def bitwise_or(self, node):
        return self.binop(node)

    def bitwise_or_1(self, node):
        return self.binop_cont(node)

    def bitwise_xor(self, node):
        return self.binop(node)

    def bitwise_xor_1(self, node):
        return self.binop_cont(node)

    def bitwise_and(self, node):
        return self.binop(node)

    def bitwise_and_1(self, node):
        return self.binop_cont(node)

    def shift_expr(self, node):
        return self.binop(node)

    def shift_expr_1(self, node):
        return self.binop_cont(node)

    def sum(self, node):
        return self.binop(node)

    def sum_1(self, node):
        return self.binop_cont(node)

    def term(self, node):
        return self.binop(node)

    def term_1(self, node):
        return self.binop_cont(node)

    def factor(self, node):
        return self.unop(node)

    def power(self, node):
        if len(node.children) == 1:
            return self.visit(node.children[0])
        lhs = self.visit(node.children[0])
        operator = self.visit(node.children[2])
        rhs = self.visit(node.children[4])
        return BinOp(lhs, operator, rhs)

    def primary_list(self, node):
        primaries = self.visit(node.children[0])
        more_primaries = self.visit_or_none(node.children[1])
        if more_primaries is not None:
            primaries = [primaries]
            primaries.extend(more_primaries)
        return primaries

    def more_primary_list(self, node):
        primaries = [self.visit(node.children[3])]
        more_primaries = self.visit_or_none(node.children[4])
        if more_primaries is not None:
            primaries.extend(more_primaries)
        return primaries

    def primary(self, node):
        ref = self.visit(node.children[0])
        access_list = self.visit_or_none(node.children[1])
        if access_list is None:
            return ref
        for access in access_list:
            if isinstance(access, CallArgs):
                ref = FuncCall(ref, access)
            elif isinstance(access, Slice):
                ref = SliceExpr(ref, access)
            elif isinstance(access, FieldAccess):
                ref = FieldAccessExpr(ref, access)
            else:
                raise InternalParserError("Unexpected access type: '{}'".format(access))
        return ref

    def access(self, node):
        access = [self.visit(node.children[1])]
        next_access = self.visit_or_none(node.children[2])
        if next_access is not None:
            access.extend(next_access)
        return access

    def field_access(self, node):
        field = self.visit(node.children[2])
        return FieldAccess(field)

    def call_args(self, node):
        args = self.visit_or_none(node.children[2])
        return CallArgs(args)

    def call_args_body(self, node):
        args = [self.visit(node.children[0])]
        more_args = self.visit_or_none(node.children[1])
        if more_args is not None:
            args.extend(more_args)
        return args

    def more_call_args_body(self, node):
        args = [self.visit(node.children[3])]
        more_args = self.visit_or_none(node.children[4])
        if more_args is not None:
            args.extend(more_args)
        return args

    def call_args_ele(self, node):
        return self.visit(node.children[0])

    def kw_arg(self, node):
        name = self.visit(node.children[0])
        value = self.visit(node.children[4])
        return KwArg(name, value)

    def pos_arg(self, node):
        value = self.visit(node.children[0])
        return PosArg(value)

    def slice(self, node):
        slice = self.visit(node.children[2])
        return Slice(slice)

    def atom(self, node):
        return self.visit(node.children[0])

    def parens(self, node):
        return self.visit(node.children[2])

    def list(self, node):
        list_elements = self.visit_or_none(node.children[2])
        if list_elements is None:
            list_elements = []
        return LiteralList(list_elements)

    def list_body(self, node):
        return self.visit(node.children[0])

    def list_elements(self, node):
        elements = [self.visit(node.children[0])]
        more_elements = self.visit_or_none(node.children[1])
        if more_elements is not None:
            elements.extend(more_elements)
        return elements

    def more_list_elements(self, node):
        if len(node.children) == 1:
            return None
        elements = [self.visit(node.children[3])]
        more_elements = self.visit_or_none(node.children[4])
        if more_elements is not None:
            elements.extend(more_elements)
        return elements

    def dict(self, node):
        dict_elements = self.visit(node.children[2])
        if dict_elements is None:
            dict_elements = dict()
        else:
            dict_elements = {k: v for k, v in dict_elements}
        return LiteralDict(dict_elements)

    def dict_body(self, node):
        if node.children[0].expr.name == "dict_elements":
            return self.visit(node.children[0])
        return None

    def dict_elements(self, node):
        elements = [self.visit(node.children[0])]
        more_elements = self.visit_or_none(node.children[1])
        if more_elements is not None:
            elements.extend(more_elements)
        return elements

    def dict_element(self, node):
        return (self.visit(node.children[0]), self.visit(node.children[4]))

    def more_dict_elements(self, node):
        if len(node.children) == 1:
            return None
        elements = [self.visit(node.children[3])]
        more_elements = self.visit_or_none(node.children[4])
        if more_elements is not None:
            elements.extend(more_elements)
        return elements

    def number(self, node):
        return self.visit(node.children[0])

    def bin(self, node):
        return LiteralInt(int(self.visit(node.children[0]), 2))

    def oct(self, node):
        return LiteralInt(int(self.visit(node.children[0]), 8))

    def hex(self, node):
        return LiteralInt(int(self.visit(node.children[0]), 16))

    def decimal(self, node):
        val = node
        if hasattr(node, "children"):
            val = self.visit(node.children[0])
        if isinstance(val, str):
            if "." in val:
                return LiteralFloat(float(val))
            else:
                return LiteralInt(int(val))
        return val

    def string(self, node):
        return self.visit(node.children[0])

    def d_ml_string(self, node): # FIXME
        val = node
        if hasattr(val, "children"):
            val = self.visit(node.children[0])
        return LiteralString(val[3:-3])

    def s_ml_string(self, node):
        val = node
        if hasattr(val, "children"):
            val = self.visit(node.children[0])
        return LiteralString(val[3:-3])

    def d_sl_string(self, node):
        val = node
        if hasattr(val, "children"):
            val = self.visit(node.children[0])
        return LiteralString(val[1:-1])

    def s_sl_string(self, node):
        val = node
        if hasattr(val, "children"):
            val = self.visit(node.children[0])
        return LiteralString(val[1:-1])

    def comp_op(self, node):
        return self.visit(node.children[0])

    def single_comp_op(self, node):
        return self.visit(node.children[0])

    def multi_comp_op(self, node):
        return self.visit(node.children[0])

    def not_in(self, node):
        return "{} {}".format(self.visit(node.children[0]), self.visit(node.children[1]))

    def in_op(self, node):
        return self.visit(node.children[0])

    def is_not(self, node):
        return "{} {}".format(self.visit(node.children[0]), self.visit(node.children[1]))

    def is_op(self, node):
        return self.visit(node.children[0])

    def has(self, node):
        return self.visit(node.children[0])

    def of(self, node):
        return self.visit(node.children[0])

    def power_op(self, node):
        return self.visit(node.children[0])

    def factor_op(self, node):
        return self.visit(node.children[0])

    def term_op(self, node):
        return self.visit(node.children[0])

    def sum_op(self, node):
        return self.visit(node.children[0])

    def shift_op(self, node):
        return self.visit(node.children[0])

    def band(self, node):
        return self.visit(node.children[0])

    def bxor(self, node):
        return self.visit(node.children[0])

    def bor(self, node):
        return self.visit(node.children[0])

    def land(self, node):
        return self.visit(node.children[0])

    def lor(self, node):
        return self.visit(node.children[0])

    def lnot(self, node):
        return self.visit(node.children[0])

    def assign_op(self, node):
        return self.visit(node.children[0])

    def dot_op(self, node):
        return self.visit(node.children[0])

    def array_op(self, node):
        return  "{}{}".format(self.visit(node.children[0]), self.visit(node.children[1]))

    def op(self, node):
        return self.visit(node.children[0])

    def name_like(self, node):
        return self.visit(node.children[0])

    def true(self, node):
        return LiteralTrue()

    def false(self, node):
        return LiteralFalse()

    def null(self, node):
        return LiteralNull()

    def name(self, node):
        val = node
        if hasattr(val, "children"):
            val = self.visit(val.children[0])
        if hasattr(val, "name"):
            val = val.name
        return NameRef(val)

    def super_ref(self, node):
        val = node
        if hasattr(val, "children"):
            val = self.visit(val.children[0])
        if hasattr(val, "name"):
            val = val.name
        return NameRef(val)

    def cls_ref(self, node):
        val = node
        if hasattr(val, "children"):
            val = self.visit(val.children[0])
        if hasattr(val, "name"):
            val = val.name
        return NameRef(val)

    def self_ref(self, node):
        val = node
        if hasattr(val, "children"):
            val = self.visit(val.children[0])
        if hasattr(val, "name"):
            val = val.name
        return NameRef(val)

    def NAME(self, node):
        return self.name(node.text)

    def D_ML_STRING(self, node):
        return self.d_ml_string(node.text)

    def S_ML_STRING(self, node):
        return self.s_ml_string(node.text)

    def D_SL_STRING(self, node):
        return self.d_sl_string(node.text)

    def S_SL_STRING(self, node):
        return self.s_sl_string(node.text)

    def BIN(self, node):
        return node.text

    def OCT(self, node):
        return node.text

    def HEX(self, node):
        return node.text

    def DECIMAL(self, node):
        return self.decimal(node.text)

    def OF(self, node):
        return node.text

    def HAS(self, node):
        return node.text

    def USING(self, node):
        return node.text

    def AS(self, node):
        return node.text

    def FN(self, node):
        return node.text

    def OPERATOR(self, node):
        return node.text

    def PUBLIC(self, node):
        return node.text

    def PRIVATE(self, node):
        return node.text

    def PROTECTED(self, node):
        return node.text

    def FINAL(self, node):
        return node.text

    def CLOSED(self, node):
        return node.text

    def CONST(self, node):
        return node.text

    def INIT(self, node):
        return node.text

    def GET(self, node):
        return node.text

    def SET(self, node):
        return node.text

    def CLASS(self, node):
        return node.text

    def TRAIT(self, node):
        return node.text

    def STATIC(self, node):
        return node.text

    def NAMESPACE(self, node):
        return node.text

    def CLS(self, node):
        return self.cls_ref(node.text)

    def SELF(self, node):
        return self.self_ref(node.text)

    def SUPER(self, node):
        return self.super_ref(node.text)

    def RETURN(self, node):
        return node.text

    def YIELD(self, node):
        return node.text

    def CONTINUE(self, node):
        return node.text

    def BREAK(self, node):
        return node.text

    def LEAVE(self, node):
        return node.text

    def DEFER(self, node):
        return node.text

    def TO(self, node):
        return node.text

    def TRUE(self, node):
        return LiteralTrue()

    def FALSE(self, node):
        return LiteralFalse()

    def NULL(self, node):
        return LiteralNull()

    def RANGE(self, node):
        return node.text

    def DOT(self, node):
        return node.text

    def TILDE(self, node):
        return node.text

    def PERCENT(self, node):
        return node.text

    def DIV_DIV(self, node):
        return node.text

    def DIV(self, node):
        return node.text

    def PLUS(self, node):
        return node.text

    def MINUS(self, node):
        return node.text

    def LSHIFT(self, node):
        return node.text

    def RSHIFT(self, node):
        return node.text

    def AMP(self, node):
        return node.text

    def CARAT(self, node):
        return node.text

    def BAR(self, node):
        return node.text

    def EQUAL_EQUAL(self, node):
        return node.text

    def BANG_EQUAL(self, node):
        return node.text

    def LE(self, node):
        return node.text

    def LT(self, node):
        return node.text

    def GE(self, node):
        return node.text

    def GT(self, node):
        return node.text

    def OR(self, node):
        return node.text

    def IS(self, node):
        return node.text

    def DOUBLE_BAR(self, node):
        return node.text

    def AND(self, node):
        return node.text

    def DOUBLE_AMP(self, node):
        return node.text

    def NOT(self, node):
        return node.text

    def BANG(self, node):
        return node.text

    def STAR(self, node):
        return node.text

    def STAR_STAR(self, node):
        return node.text

    def COLON(self, node):
        return node.text

    def EQUAL(self, node):
        return node.text

    def IF(self, node):
        return node.text

    def ELIF(self, node):
        return node.text

    def ELSE(self, node):
        return node.text

    def FOR(self, node):
        return node.text

    def ELFOR(self, node):
        return node.text

    def IN(self, node):
        return node.text

    def WHILE(self, node):
        return node.text

    def ELWHILE(self, node):
        return node.text

    def OPEN_PAREN(self, node):
        return node.text

    def CLOSE_PAREN(self, node):
        return node.text

    def OPEN_SQUARE(self, node):
        return node.text

    def CLOSE_SQUARE(self, node):
        return node.text

    def OPEN_BRACE(self, node):
        return node.text

    def CLOSE_BRACE(self, node):
        return node.text

    def COMMA(self, node):
        return node.text

    def __default__(self, node):
        print("'{}'".format(node.name if hasattr(node, "name") else node.expr_name))
        return node

    def visit_or_none(self, node):
        if len(node.children) > 0:
            return self.visit(node.children[0])
        return None

    def visit(self, node):
        if len(node.children) == 1 and node.children[0].expr.name == '':
            node.children = node.children[0].children
        if len(node.expr.name) < 1:
            raise ValueError("Empty node name!")
        if hasattr(self, node.expr.name):
            callee = node.expr.name
            return getattr(self, callee)(node)
        return self.__default__(node)

class PpegParser(BaseParser):
    def __init__(self, grammar_file=DEFAULT_GRAMMAR_PATH):
        super().__init__(grammar_file)
        self.parser = Grammar(self.grammar_text)
        self.ast_gen = ASTGen()

    def parse(self, text, print_tree=False):
        parse_res = self.parser.parse(text)
        if print_tree is True:
            print(parse_res)
        tree_ast = self.ast_gen.visit(parse_res)
        if print_tree is True:
            print("AST")
            print("")
            print(tree_ast)
        return tree_ast

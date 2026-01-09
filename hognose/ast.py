import copy
from .utils import format_or_empty
from .scope import Scope

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

class LiteralNull:
    def __init__(self):
        pass

    def __str__(self):
        return "LiteralNull"

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        return None

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
            if isinstance(target, (list, dict, str)): # FIXME
                setattr(target, field_name, assign_val)
            else:
                target = target.assign_field(field_name, assign_val)
        if isinstance(target, (list, dict, str)): # FIXME
            return getattr(target, field_name)
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
    def __init__(self, args=None):
        self.pos_args = []
        self.kw_args = {}
        if args is not None:
            for arg in args:
                if isinstance(arg, PosArg):
                    self.pos_args.append(arg.value)
                elif isinstance(arg, KwArg):
                    if arg.name in self.kw_args:
                        raise ValueError("Duplicate kwarg name '{}'".format(arg.name))
                    self.kw_args[arg.name] = arg.value

    def __str__(self):
        return "CallArgs: {}".format(", ".join([str(x) for x in self.pos_args] + ["{}={}".format(k, v) for k, v in self.kw_args.items()]))

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
        callee = self.callee.eval(symbol_table)
        if type(callee).__name__ == "builtin_function_or_method": # FIXME
            callee = BuiltinFnDef(callee)
        return callee.eval(symbol_table, self.args.pos_args, self.args.kw_args)

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
    def __init__(self, exit_type, exit_val=None, exit_dir=None):
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
    def __init__(self, target, operator, value, type_expr=None):
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
    def __init__(self, guard, body, elexpr=None):
        self.guard = guard
        self.body = body
        self.elexpr = elexpr

    def __str__(self):
        return "If: {} then {}{}".format(self.guard, self.body,
                " else {}".format(self.elexpr) if self.elexpr is not None else "")

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
        if self.elexpr is not None:
            return self.elexpr.eval(symbol_table)

class WhileExpr:
    def __init__(self, guard, body, elexpr=None):
        self.guard = guard
        self.body = body
        self.elexpr = elexpr

    def __str__(self):
        return "If: {} then {}{}".format(self.guard, self.body,
                " else {}".format(self.elexpr) if self.elexpr is not None else "")

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
        elif self.elexpr is not None:
            return self.elexpr.eval(symbol_table)

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
    def __init__(self, name, args, body, type_expr=None):
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

class OpDeclExpr:
    def __init__(self, operator, args, body, type_expr=None):
        self.operator = operator
        self.args = args
        self.body = body
        self.type_expr = type_expr

    def __str__(self):
        return "OpDecl: {}({}) {}".format(self.name if self.name is not None else "", self.args if self.args is not None else "", self.body)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, properties=None, class_decl=False):
        NotImplementedError

class ForExpr:
    def __init__(self, induction_var, induction_expr, body, elexpr=None):
        self.induction_var = induction_var
        self.induction_expr = induction_expr
        self.body = body
        self.elexpr = elexpr

    def __str__(self):
        return "For: {} in {} do {}".format(self.induction_var,
                                            self.induction_expr,
                                            self.body)
    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, class_decl=False):
        at_least_once = False
        last_val = None
        induction_var = self.induction_var
        if hasattr(induction_var, "expr"):
            induction_var = induction_var.expr
        induction_var = induction_var.name # FIXME
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
        elif self.elexpr is not None:
            return self.elexpr.eval(symbol_table)

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
        return "ObjDef: {} {} {} {{{}}}".format(self.obj_type, self.obj_name, [str(x) for x in self.parents] if self.parents is not None else "",
                        ", ".join(["{}".format(name) for name, _ in self.members.symbols.items()])
                )

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

class ClassParentDecl:
    def __init__(self, expr, assignment=None):
        self.expr = expr
        self.assignment = assignment

    def __str__(self):
        return "ClassParentDecl: {}{}".format(self.expr, format_or_empty("={}", self.assignment))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        if self.assignment is not None:
            raise NotImplementedError
        elif not (isinstance(self.expr, NameRef) or isinstance(self.expr, ClassDecl)):
            raise NotImplementedError
        return self.expr.eval(symbol_table)

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

class NamespaceDecl:
    def __init__(self, name=None, body=None):
        self.name = name
        self.body = body
        self.whole_file = False
        if self.body is None:
            self.whole_file = True

    def __str__(self):
        return "NamespaceDecl:{}{}{}".format(
            format_or_empty(" {}", self.name),
            " (whole file)" if self.whole_file else "",
            format_or_empty(" {}", self.body))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        raise NotImplementedError

class DoubleStarExpr:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return "DoubleStarExpr: **{}".format(self.expr)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        raise NotImplementedError

class StarExpr:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return "StarExpr: *{}".format(self.expr)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        raise NotImplementedError

class UsingExpr:
    def __init__(self, target, as_expr=None):
        self.target = target
        self.as_expr = as_expr

    def __str__(self):
        return "UsingExpr: {}{}".format(self.target, format_or_empty(" as {}", self.as_expr))

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table):
        raise NotImplementedError

class BlockExpr:
    def __init__(self, exprs):
        self.exprs = exprs

    def __str__(self):
        return "BlockExpr: {}".format(self.exprs)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, class_decl=False, return_scope=False):
        return self.exprs.eval(symbol_table, class_decl=class_decl, return_scope=return_scope)

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

class LabeledExpr:
    def __init__(self, label, expr):
        self.label = label
        self.expr = expr

    def __str__(self):
        return "{}: {}".format(self.label, self.expr)

    def __repr__(self):
        return self.__str__()

    def eval(self, symbol_table, class_decl=False, return_scope=False):
        return NotImplementedError

class Expr:
    def __init__(self, expr, properties=None, type_expr=None):
        self.expr = expr
        self.properties = properties
        self.type_expr = type_expr

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


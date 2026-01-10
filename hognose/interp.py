import sys
import inspect
from .scope import Scope
from .ast import BuiltinFnDef, ObjDef
from .parser import BaseParser
from .parsers import parsers

class Interpreter:
    def __init__(self, parser=None, **kwargs):
        if isinstance(parser, BaseParser):
            self.parser = parser
        elif isinstance(parser, str):
            if parser in parsers:
                self.parser = parsers[parser](**kwargs)
            else:
                raise ValueError("No parser '{}'. Choose from: {}".format(parser, list(parsers.keys())))
        elif inspect.isclass(parser) and issubclass(parser, BaseParser):
            self.parser = parser(**kwargs)
        elif parser is None:
            self.parser = parsers["ppeg"](**kwargs)
        else:
            raise ValueError("Invalid parser '{}' of type '{}'".format(parser, type(parser).__name__))

        self.scope = Scope(symbols={
            "print": BuiltinFnDef(print),
            "int": BuiltinFnDef(int),
            "float": BuiltinFnDef(float),
            "len": BuiltinFnDef(len)
        })

    def eval(self, text, print_tree=False):
        self.scope = self.parser.parse(text, print_tree=print_tree).eval(self.scope, return_scope=True)

    def eval_from_file(self, file_path, print_input=False, print_tree=False, commandline_args=sys.argv):
        input_text = None
        with open(file_path) as f:
            input_text = f.read()
        if print_input:
            for line_no, line in enumerate(input_text.splitlines()):
                print("{}: {}".format(line_no + 1, line))
        if "env" not in self.scope.symbols:
            self.scope.assign("env", ObjDef("namespace",
                members=Scope(),
                obj_name="env")
            )
        self.scope.symbols["env"].members.assign("args", commandline_args, immediate=True)
        self.eval(input_text, print_tree=print_tree)

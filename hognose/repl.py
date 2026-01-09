from cmd import Cmd
from .interp import Interpreter
from .scope import MissingSymbolError
from .ast import ArgumentError
from .parser import ParserError

class Repl(Cmd):
    def __init__(self, **kwargs):
        super().__init__()
        self.__interp = Interpreter(**kwargs)
        self.intro = "\n".join([
        "Welcome to Hognose. This language is very under development.",
        "Type 'quit' or 'exit' to exit."
        ])
        self.__normal_prompt = "~~> "
        self.__oops_prompt = "~!> "
        self.prompt = self.__normal_prompt

    def default(self, line):
        try:
            self.__interp.eval(line + ";")
            self.prompt = self.__normal_prompt
        except (ParserError, ArgumentError, MissingSymbolError) as e:
            self.prompt = self.__oops_prompt
            print(e)

    def completenames(self, text, *args):
        path = None
        if "." in text:
            path = text.split(".")
        ret_names = []
        symbol_names = self.__interp.scope.symbol_names()
        if path is None:
            if text in "exit":
                ret_names.append("exit")
            if text in "quit":
                ret_names.append("quit")
            if text in "help":
                ret_names.append("help")
            for symbol in symbol_names:
                if symbol.startswith(text):
                    ret_names.append(symbol)
        if path[0] in symbol_names:
            root_symbol = self.__interp.scope.get(path[0])
            ret_names = []
            ret_path = [path[0]]
            members = root_symbol.members.symbol_names(immediate=True) if hasattr(root_symbol, "members") else []
            for path_ele in path[1:-1]:
                if path_ele in members:
                    root_symbol = root_symbol.members.get(path_ele)
                    members = root_symbol.members.symbol_names(immediate=True) if hasattr(root_symbol, "members") else []
                else:
                    members = []
                    break
            for symbol in members:
                if symbol.startswith(path[-1]):
                    ret_names.append(".".join([*ret_path, symbol]))
        return ret_names

    def do_help(self, arg):
        path = None
        if "." in arg:
            path = arg.split(".")
        else:
            path = [arg]
        if len(arg) == 0:
            self.stdout.write("help <cmd>: choose from '{}'\n".format(["exit", "quit", *self.__interp.scope.symbol_names()]))
            return
        if arg in ["exit", "quit"]:
            self.stdout.write("{}: Exit the REPL\n".format(arg))
            return
        elif arg in ["help"]:
            self.stdout.write("{}: This\n".format(arg))
            return
        elif path[0] in self.__interp.scope.symbol_names():
            symbol = self.__interp.scope.get(path[0])
            members = symbol.members.symbol_names(immediate=True) if hasattr(symbol, "members") else []
            for path_ele in path[1:]:
                if path_ele in members:
                    symbol = symbol.members.get(path_ele)
                    members = symbol.members.symbol_names(immediate=True) if hasattr(symbol, "members") else []
                else:
                    symbol = None
                    break
            if symbol is not None:
                self.stdout.write("{}: Hognose object: '{}'\n".format(arg, symbol))
                return
        self.stdout.write("Unknown symbol '{}'\n".format(arg))

    def emptyline(self):
        pass

    def do_exit(self, line):
        return True

    def do_quit(self, line):
        return True

    def cmdloop(self):
        try:
            super().cmdloop()
        except KeyboardInterrupt:
            print("")


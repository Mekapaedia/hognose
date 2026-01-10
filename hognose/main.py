import argparse
from .interp import Interpreter
from .repl import Repl

def str_to_bool(text):
    if isinstance(text, bool):
        return text
    elif isinstance(text, str):
        text = text.strip().lower()
        if text == "true":
            return True
        elif text == "false":
            return False
    raise ValueError("Invalid value for conversion to boolean: '{}'".format(text))

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--print-input", nargs='?', const=True, default=False, type=str_to_bool, help="Whether to print the input file")
    parser.add_argument("--print-tree", nargs='?', const=True, default=False, type=str_to_bool, help="Whether to print the parse tree")
    parser.add_argument("input_file", nargs='?', help="Input Hognose file. If not specified, will drop to REPL")
    parser.add_argument("command_line_args", nargs="*", help="Command line arguments to script")
    return parser

def main():
    args_parser = get_arg_parser()
    args = args_parser.parse_args()
    if args.input_file is not None:
        Interpreter().eval_from_file(args.input_file, print_input=args.print_input, print_tree=args.print_tree, commandline_args=args.command_line_args)
    else:
        Repl().cmdloop()
    return 0

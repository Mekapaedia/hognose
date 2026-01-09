class ParserError(Exception):
    pass

class InternalParserError(Exception):
    pass

class BaseParser:
    def __init__(self, grammar_file):
        self.grammar_text = None
        with open(grammar_file) as f:
            self.grammar_text = f.read()

    def parse(self, text, print_tree=False):
        raise NotImplementedError

    def parse_from_file(self, file_path, print_input=False, print_tree=False):
        input_text = None
        with open(file_path) as f:
            input_text = f.read()
        if print_input:
            for line_no, line in enumerate(input_text.splitlines()):
                print("{}: {}".format(line_no + 1, line))
        return self.parse(input_text, print_tree=print_tree)

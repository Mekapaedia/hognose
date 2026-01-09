def _basic_tree_print(tree):
    if hasattr(tree, "pretty"):
        print(tree.pretty())
    else:
        print(tree)

tree_print = _basic_tree_print
try:
    import rich
    def _rich_tree_print(tree):
        rich.print(tree)
    tree_print = _rich_tree_print
except ImportError:
    pass

def format_or_empty(string, nullable):
    return string.format(nullable) if nullable is not None else ""


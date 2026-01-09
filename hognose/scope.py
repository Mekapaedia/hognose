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

    def symbol_names(self, immediate=False):
        symbol_names = list(self.symbols.keys())
        if self.captured_scope is not None:
            symbol_names.extend(self.captured_scope.symbol_names())
        if self.parent_scope is not None and immediate is False:
            symbol_names.extend(self.parent_scope.symbol_names())
        return symbol_names

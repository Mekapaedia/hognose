Program                =     Statement* EOF
Statement              =     Expression StatementEnd
StatementEnd           =     SEMI_COLON

Expression             =     FunctionDefinition
                       /     WhileExpression
                       /     ForExpression
                       /     IfExpression
                       /     VariableBind

VariableBind           =     VariableAccess (COMMA VariableAccess)* (AssignOp Expression)?
VariableAccess         =     Atom (AccessSpecifier)* (COLON TypeSpecification)?
AccessSpecifier        =     ArrayAccess
                       /     FieldAccess
                       /     FunctionCall
Atom                   =     AssociativeDefinition
                       /     ArrayDefinition
                       /     NAME
AssociativeDefinition  =     OPEN_BRACE (VariableAccess COLON Expression)? (COMMA VariableAccess COLON Expression)* CLOSE_BRACE
ArrayDefinition        =     OPEN_SQUARE Expression? (COMMA Expression)* CLOSE_SQUARE

TypeSpecification      =     TypeExpression (COMMA TypeExpression)*
TypeExpression         =     AssociativeType (TypeSpecOp AssociativeType)*
AssociativeType        =     OPEN_BRACE TypeSpecification COLON TypeSpecification CLOSE_BRACE
                       /     ArrayType
ArrayType              =     OPEN_SQUARE TypeSpecification CLOSE_SQUARE
                       /     TypeReference
TypeReference          =     LEFT_CARAT VariableAccess RIGHT_CARAT
                       /     SingularType
SingularType           =     NAME
                       /     ANY_TYPE
                       /     NEVER_TYPE
TypeSpecOp             =     PLUS
                       /     MINUS

AssignOp               =     EQUAL

ArrayAccess            =     OPEN_SQUARE Expression CLOSE_SQUARE
FieldAccess            =     DOT NAME
FunctionCall           =     OPEN_PAREN VariableBind* CLOSE_PAREN

Block                  =     OPEN_BRACE Statement* CLOSE_BRACE
FunctionDefinition     =     FUNC NAME? OPEN_PAREN VariableBind* CLOSE_PAREN (COLON TypeSpecification)? Block?
WhileExpression        =     WHILE OPEN_PAREN Expression CLOSE_PAREN Block
ForExpression          =     FOR OPEN_PAREN VariableAccess IN Expression CLOSE_PAREN Block
IfExpression           =     IF OPEN_PAREN Expression CLOSE_PAREN Block (ELIF OPEN_PAREN Expression CLOSE_PAREN Block)* (ELSE Block)?

NAME         = r'[a-zA-Z][a-zA-Z_\-0-9]*'

ANY_TYPE     = "Any"
NEVER_TYPE   = "None"
FUNC         = "fn"
IF           = "if"
ELIF         = "elif"
ELSE         = "else"
WHILE        = "while"
FOR          = "for"
IN           = "in"
IS           = "is"
TYPE         = "type"
DEFER        = "defer"
AND          = "and"
OR           = "or"
NOT          = "not"
AS           = "as"

PLUS         = "+"
MINUS        = "-"
STAR         = "*"
FWD_SLASH    = "/"
BACK_SLASH   = "\\"
UNDERSCORE   = "_"
EQUAL        = "="
DOT          = "."
COMMA        = ","
COLON        = ":"
SEMI_COLON   = ";"
OPEN_BRACE   = "{"
CLOSE_BRACE  = "}"
OPEN_SQUARE  = "["
CLOSE_SQUARE = "]"
OPEN_PAREN   = "("
CLOSE_PAREN  = ")"
LEFT_CARAT   = "<"
RIGHT_CARAT  = ">"

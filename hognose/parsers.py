parsers = {}
def add_parser(parser):
    parsers[parser.__name__.replace("Parser", "").lower()] = parser

from .lark.parse import LarkParser
from .ppeg.parse import PpegParser

add_parser(LarkParser)
add_parser(PpegParser)

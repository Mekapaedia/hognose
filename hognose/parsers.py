parsers = {}
def add_parser(parser):
    parsers[parser.__name__.replace("Parser", "").lower()] = parser

from .ppeg.parse import PpegParser

add_parser(PpegParser)

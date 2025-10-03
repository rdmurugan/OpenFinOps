"""
Vizly Native Rendering System
Pure Python rendering backend replacing matplotlib dependency.
"""

from .canvas import Canvas, Color, Point, Rectangle, Line
from .renderer import PureRenderer, ImageRenderer
from .primitives import Shape, Path, Text, Image
from .export import PNGExporter, SVGExporter, PDFExporter

__all__ = [
    "Canvas",
    "Color",
    "Point",
    "Rectangle",
    "Line",
    "PureRenderer",
    "ImageRenderer",
    "Shape",
    "Path",
    "Text",
    "Image",
    "PNGExporter",
    "SVGExporter",
    "PDFExporter",
]

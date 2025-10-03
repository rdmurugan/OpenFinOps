"""
Fine-Grained Control API
========================

Advanced control over every aspect of chart appearance and behavior,
providing matplotlib-level customization capabilities.
"""

from .axes_control import *
from .styling_control import *
from .layout_control import *

__all__ = [
    # Axes Control
    'Axes',
    'AxisControl',
    'GridControl',
    'TickLocator',
    'LinearLocator',
    'MultipleLocator',
    'LogLocator',
    'TickFormatter',
    'ScalarFormatter',
    'LogFormatter',
    'TickProperties',
    'SpineProperties',
    'GridProperties',

    # Styling Control
    'StyleManager',
    'LegendControl',
    'TextControl',
    'ColorControl',
    'LineStyleControl',
    'MarkerControl',
    'ColorPalette',
    'MarkerStyle',
    'TextStyle',

    # Layout Control
    'LayoutManager',
    'SubplotGrid',
    'FigureManager',
    'SubplotSpec',
    'LayoutGeometry',
    'create_subplot_grid',
    'create_figure',
    'tight_layout',
]
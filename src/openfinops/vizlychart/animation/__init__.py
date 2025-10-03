"""
Animation System for OpenFinOps
===============================

Advanced animation capabilities for creating dynamic visualizations,
time series animations, and interactive charts.
"""

from .animation_core import *

__all__ = [
    'Animation',
    'AnimationFrame',
    'create_gif_animation',
    'animate_chart',
]
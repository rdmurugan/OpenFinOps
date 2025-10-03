"""
Scientific Visualization Tools
==============================

Advanced scientific and technical visualization capabilities including
statistical plots, signal processing, and domain-specific visualizations.
"""

from .statistics import *
from .signal_processing import *
from .specialized_plots import *

__all__ = [
    'qqplot',
    'residual_plot',
    'correlation_matrix',
    'pca_plot',
    'dendrogram',
    'spectrogram',
    'phase_plot',
    'bode_plot',
    'nyquist_plot',
    'waterfall_plot',
    'parallel_coordinates',
]
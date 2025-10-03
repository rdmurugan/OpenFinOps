"""
AI-Powered Visualization Module
==============================

Natural language interfaces and intelligent chart generation.
"""

from .nlp import AIChartGenerator, NaturalLanguageProcessor, create, ChartIntent
from .smart_selection import SmartChartSelector, recommend_chart, DataProfile, ChartRecommendation
from .styling import NaturalLanguageStylist, style_chart, parse_style, StyleConfig

__all__ = [
    'AIChartGenerator',
    'NaturalLanguageProcessor',
    'create',
    'ChartIntent',
    'SmartChartSelector',
    'recommend_chart',
    'DataProfile',
    'ChartRecommendation',
    'NaturalLanguageStylist',
    'style_chart',
    'parse_style',
    'StyleConfig'
]
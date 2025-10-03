"""
Vizly Engineering Module
=======================

Advanced engineering analysis and visualization capabilities including:
- Finite Element Analysis (FEA)
- Computational Fluid Dynamics (CFD)
- Modal Analysis and Vibration
- Thermal Analysis
- CAD Model Import and Rendering

This module provides GPU-accelerated solvers and specialized chart types
for engineering applications.
"""

from .bode import BodePlot
from .stress import StressStrainChart
from .fem import FEMSolver, FEMChart
from .cfd import CFDSolver, CFDChart
from .modal import ModalSolver, ModalChart
from .thermal import ThermalSolver, ThermalChart
from .cad_import import IGESImporter, STEPImporter, STLImporter, CADViewer

__all__ = [
    "BodePlot",
    "StressStrainChart",
    'FEMSolver', 'FEMChart',
    'CFDSolver', 'CFDChart',
    'ModalSolver', 'ModalChart',
    'ThermalSolver', 'ThermalChart',
    'IGESImporter', 'STEPImporter', 'STLImporter', 'CADViewer'
]

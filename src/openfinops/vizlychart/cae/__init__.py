"""Computer-Aided Engineering (CAE) visualization tools for Vizly."""

from .mesh import MeshRenderer, FEAMesh, StructuralMesh
from .fields import ScalarField, VectorField, TensorField
from .analysis import StressAnalysis, ThermalAnalysis, ModalAnalysis
from .geometry import CADGeometry, GeometryLoader
from .animation import TimeSeriesAnimation, DeformationAnimation

__all__ = [
    "MeshRenderer",
    "FEAMesh",
    "StructuralMesh",
    "ScalarField",
    "VectorField",
    "TensorField",
    "StressAnalysis",
    "ThermalAnalysis",
    "ModalAnalysis",
    "CADGeometry",
    "GeometryLoader",
    "TimeSeriesAnimation",
    "DeformationAnimation",
]

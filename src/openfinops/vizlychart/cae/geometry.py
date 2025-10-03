"""Geometry utilities for CAE visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryData:
    """Container for geometry data."""

    vertices: np.ndarray
    faces: np.ndarray
    name: str
    properties: dict


class CADGeometry:
    """CAD geometry representation."""

    def __init__(self, name: str = "Geometry") -> None:
        self.name = name
        self.geometries: List[GeometryData] = []

    def add_geometry(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        name: Optional[str] = None,
        **properties,
    ) -> None:
        """Add geometry data."""
        geom_name = name or f"Geometry_{len(self.geometries)}"
        geometry = GeometryData(
            vertices=vertices, faces=faces, name=geom_name, properties=properties
        )
        self.geometries.append(geometry)


class GeometryLoader:
    """Geometry file loader."""

    @staticmethod
    def load_stl(file_path: str) -> CADGeometry:
        """Load STL file."""
        logger.info(f"Loading STL file: {file_path}")
        # Placeholder implementation
        geometry = CADGeometry("STL_Geometry")
        return geometry

    @staticmethod
    def load_obj(file_path: str) -> CADGeometry:
        """Load OBJ file."""
        logger.info(f"Loading OBJ file: {file_path}")
        # Placeholder implementation
        geometry = CADGeometry("OBJ_Geometry")
        return geometry

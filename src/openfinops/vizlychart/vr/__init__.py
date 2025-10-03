"""
Vizly VR/AR Visualization Module
===============================

Advanced virtual and augmented reality visualization capabilities for Vizly.
Supports WebXR, OpenXR, and immersive spatial visualization.

Basic Usage:
    >>> import vizly.vr as vr
    >>> # WebXR session
    >>> session = vr.WebXRSession("immersive-vr")
    >>> session.add_chart({'type': 'scatter', 'data': data})
    >>> await session.request_session()

    >>> # VR scene
    >>> scene = vr.VRScene()
    >>> chart = vizly.SurfaceChart()
    >>> vr_chart = scene.add_chart(chart)
    >>> scene.start_vr_session()

Supported Platforms:
    - WebXR (browser-based VR/AR)
    - Native VR/AR applications
    - Spatial computing environments
    - Immersive visualization displays
"""

from .core import VRScene, ARScene, XREnvironment, VRChart, ARChart
from .webxr import WebXRSession, WebXRServer
from .spatial import SpatialRenderer, VRCanvas, ARCanvas, SpatialObject
from .immersive_charts import (
    ImmersiveChart, VRScatterChart, VRSurfaceChart, VRLineChart,
    AROverlayChart, ImmersiveChartRenderer
)

__version__ = "0.5.0"
__all__ = [
    # Core VR/AR
    "VRScene", "ARScene", "XREnvironment", "VRChart", "ARChart",
    # WebXR
    "WebXRSession", "WebXRServer",
    # Spatial rendering
    "SpatialRenderer", "VRCanvas", "ARCanvas", "SpatialObject",
    # Immersive charts
    "ImmersiveChart", "VRScatterChart", "VRSurfaceChart", "VRLineChart",
    "AROverlayChart", "ImmersiveChartRenderer",
]
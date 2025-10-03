"""
Vizly Enterprise Module
=======================

Enterprise-grade visualization and analytics platform with advanced security,
performance, and integration capabilities.

Key Features:
- Enterprise security and compliance
- Advanced GIS and geospatial analytics
- High-performance big data processing
- AI-powered visualization assistance
- Enterprise system integration
- Collaboration and sharing platform
"""

# Core Security Infrastructure
from .security import EnterpriseSecurityManager, ComplianceAuditLogger
from .admin import UserManager, RoleManager, AuditManager
from .licensing import LicenseManager, LicenseEnforcer
from .benchmarks import PerformanceBenchmark

# Enterprise Charts & Visualization
from .charts import (
    EnterpriseBaseChart, ExecutiveDashboardChart, FinancialAnalyticsChart,
    ComplianceChart, RiskAnalysisChart, EnterpriseChartFactory
)
from .themes import (
    EnterpriseTheme, PresentationTheme, PrintTheme, DarkTheme,
    ThemeManager, BrandingConfig, AccessibilityConfig
)
from .exports import EnterpriseExporter, ExportConfig, ReportSection

# Enterprise Server & API
try:
    from .server import EnterpriseServer
except ImportError:
    # Fallback if aiohttp not available
    class EnterpriseServer:
        def __init__(self, *args, **kwargs):
            raise ImportError("aiohttp is required for enterprise server. Install with: pip install aiohttp")

# GIS & Analytics
try:
    from .gis import EnterpriseGISEngine, SpatialAnalyticsEngine, RealTimeTracker
    from .performance import DistributedDataEngine, GPUAcceleratedRenderer, IntelligentDataSampler, EnterprisePerformanceBenchmark
except ImportError:
    # Placeholder classes for missing modules
    class EnterpriseGISEngine:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("GIS features will be implemented in Phase 2")

    class SpatialAnalyticsEngine:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Spatial analytics will be implemented in Phase 2")

    class DistributedDataEngine:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Distributed computing will be implemented in Phase 2")

    class GPUAcceleratedRenderer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("GPU acceleration will be implemented in Phase 2")

# Collaboration & AI (future phases)
try:
    from .collaboration import WorkspaceManager, VisualizationVersionControl
    from .connectors import EnterpriseDataConnectors
    from .ai import VizlyAI, ChartRecommendationEngine
except ImportError:
    # Placeholder classes for future implementation
    class WorkspaceManager:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Collaboration features will be implemented in Phase 2B")

    class VisualizationVersionControl:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Version control will be implemented in Phase 2B")

    class EnterpriseDataConnectors:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Data connectors will be implemented in Phase 2A")

    class VizlyAI:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AI features will be implemented in future phases")

    class ChartRecommendationEngine:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AI recommendations will be implemented in future phases")

__all__ = [
    # Core Security Infrastructure
    "EnterpriseSecurityManager",
    "ComplianceAuditLogger",
    "UserManager",
    "RoleManager",
    "AuditManager",
    "LicenseManager",
    "LicenseEnforcer",

    # Enterprise Charts & Visualization
    "EnterpriseBaseChart",
    "ExecutiveDashboardChart",
    "FinancialAnalyticsChart",
    "ComplianceChart",
    "RiskAnalysisChart",
    "EnterpriseChartFactory",

    # Themes & Styling
    "EnterpriseTheme",
    "PresentationTheme",
    "PrintTheme",
    "DarkTheme",
    "ThemeManager",
    "BrandingConfig",
    "AccessibilityConfig",

    # Export & Reporting
    "EnterpriseExporter",
    "ExportConfig",
    "ReportSection",

    # Enterprise Server & API
    "EnterpriseServer",

    # Performance & Benchmarking
    "PerformanceBenchmark",
    "DistributedDataEngine",
    "GPUAcceleratedRenderer",
    "IntelligentDataSampler",
    "EnterprisePerformanceBenchmark",

    # GIS & Geospatial
    "EnterpriseGISEngine",
    "SpatialAnalyticsEngine",
    "RealTimeTracker",

    # Collaboration
    "WorkspaceManager",
    "VisualizationVersionControl",

    # Data Integration
    "EnterpriseDataConnectors",

    # AI & Analytics
    "VizlyAI",
    "ChartRecommendationEngine",
]

__version__ = "1.0.0-enterprise"
__enterprise_license__ = "Commercial - Enterprise License Required"
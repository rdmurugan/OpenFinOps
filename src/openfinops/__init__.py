"""
OpenFinOps - Open Source FinOps Platform for AI/ML Cost Observability
=====================================================================

A comprehensive platform for tracking, analyzing, and optimizing costs across
AI/ML infrastructure and operations.

Features:
- LLM Training Cost Monitoring
- RAG Pipeline Analytics
- Multi-Cloud Cost Tracking (AWS, Azure, GCP)
- AI API Usage Tracking (OpenAI, Anthropic)
- Executive Dashboards (CFO, COO, Infrastructure)
- Cost Attribution and Reporting
- AI-Powered Optimization Recommendations

License: Apache 2.0
"""

__version__ = "0.1.2"
__author__ = "OpenFinOps Contributors"
__license__ = "Apache-2.0"

# Import main components for easy access
from openfinops.observability.observability_hub import ObservabilityHub
from openfinops.observability.llm_observability import LLMObservabilityHub
from openfinops.observability.finops_dashboards import LLMFinOpsDashboardCreator
from openfinops.observability.cost_observatory import CostObservatory

__all__ = [
    "ObservabilityHub",
    "LLMObservabilityHub",
    "LLMFinOpsDashboardCreator",
    "CostObservatory",
    "__version__",
]


def get_version():
    """Return the version of OpenFinOps."""
    return __version__


def hello():
    """Print a welcome message."""
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║               🌟 Welcome to OpenFinOps 🌟                 ║
    ║                                                           ║
    ║   Open Source FinOps Platform for AI/ML Observability    ║
    ║                    Version {__version__}                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝

    Quick Start:
    ------------
    from openfinops import ObservabilityHub, LLMObservabilityHub

    hub = ObservabilityHub()
    llm_hub = LLMObservabilityHub()

    For more information, visit: https://github.com/openfinops/openfinops
    """)

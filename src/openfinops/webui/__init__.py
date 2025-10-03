"""
OpenFinOps Web UI Module
=========================

Modern web interface for OpenFinOps with real-time updates and beautiful visualizations.
"""

__all__ = ["app", "start_server"]

try:
    from .server import app, start_server
except ImportError:
    app = None
    start_server = None

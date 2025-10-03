"""
Backend Management System
========================

Unified interface for switching between different visualization backends.
"""

from .unified import (
    BackendType,
    BackendCapabilities,
    UnifiedBackend,
    get_backend,
    set_backend,
    list_backends,
    get_capabilities
)

__all__ = [
    'BackendType',
    'BackendCapabilities',
    'UnifiedBackend',
    'get_backend',
    'set_backend',
    'list_backends',
    'get_capabilities'
]
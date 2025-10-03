"""
OpenFinOps Dashboard Module
============================

Executive and role-based dashboards for FinOps visibility.

Available Dashboards:
- CFODashboard: Financial executive dashboard
- COODashboard: Operations executive dashboard
- InfrastructureLeaderDashboard: Infrastructure team dashboard
- FinanceAnalystDashboard: Finance analyst dashboard
- ModelCostDashboard: AI/ML model cost tracking
"""

__all__ = [
    "CFODashboard",
    "COODashboard",
    "InfrastructureLeaderDashboard",
    "FinanceAnalystDashboard",
    "ModelCostDashboard",
    "DashboardRouter",
]

# Import dashboard classes
try:
    from .cfo_dashboard import CFODashboard
except ImportError:
    CFODashboard = None

try:
    from .coo_dashboard import COODashboard
except ImportError:
    COODashboard = None

try:
    from .infrastructure_leader_dashboard import InfrastructureLeaderDashboard
except ImportError:
    InfrastructureLeaderDashboard = None

try:
    from .finance_analyst_dashboard import FinanceAnalystDashboard
except ImportError:
    FinanceAnalystDashboard = None

try:
    from .model_cost_dashboard import ModelCostDashboard
except ImportError:
    ModelCostDashboard = None

try:
    from .dashboard_router import DashboardRouter
except ImportError:
    DashboardRouter = None

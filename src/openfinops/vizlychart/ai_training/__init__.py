"""
AI Training Visualization System for OpenFinOps
==============================================

Real-time, interactive visualization tools for AI/ML model training monitoring,
early stopping analysis, and hyperparameter optimization visualization.

Key Features:
- Real-time training metrics streaming and visualization
- Interactive early stopping analysis with threshold controls
- Multi-model comparison dashboards
- Hyperparameter optimization 3D/4D visualization
- Training anomaly detection and alerting
- GPU utilization and resource monitoring
- Gradient flow visualization
- Loss landscape exploration
- Model architecture visualization

Usage:
    >>> import openfinops as vc
    >>> from openfinops.ai_training import TrainingMonitor
    >>>
    >>> # Real-time training visualization
    >>> monitor = TrainingMonitor()
    >>> monitor.log_metrics(epoch=1, loss=0.5, accuracy=0.8, lr=0.001)
    >>> monitor.show_realtime_dashboard()
    >>>
    >>> # Early stopping analysis
    >>> early_stop = vc.EarlyStoppingAnalyzer()
    >>> early_stop.analyze_training_curve(losses, patience=10)
    >>> early_stop.show_interactive_analysis()
"""

from .training_monitor import TrainingMonitor, RealTimeTrainingDashboard
from .early_stopping import EarlyStoppingAnalyzer
from .hyperparameter_viz import HyperparameterOptimizer3D, ParamSpaceExplorer
from .model_comparison import ModelComparisonDashboard, TrainingComparator
from .streaming_metrics import MetricsStreamer, LiveMetricsServer
from .anomaly_detection import TrainingAnomalyDetector, AnomalyAlertSystem
from .gradient_viz import GradientFlowVisualizer, WeightDistributionTracker
from .loss_landscape import LossLandscapeExplorer, LossContourVisualizer
from .resource_monitor import GPUMonitor, ResourceDashboard

__all__ = [
    # Core Training Monitoring
    "TrainingMonitor",
    "RealTimeTrainingDashboard",

    # Early Stopping Analysis
    "EarlyStoppingAnalyzer",

    # Hyperparameter Optimization
    "HyperparameterOptimizer3D",
    "ParamSpaceExplorer",

    # Model Comparison
    "ModelComparisonDashboard",
    "TrainingComparator",

    # Streaming & Real-time
    "MetricsStreamer",
    "LiveMetricsServer",

    # Anomaly Detection
    "TrainingAnomalyDetector",
    "AnomalyAlertSystem",

    # Advanced Analysis
    "GradientFlowVisualizer",
    "WeightDistributionTracker",
    "LossLandscapeExplorer",
    "LossContourVisualizer",

    # Resource Monitoring
    "GPUMonitor",
    "ResourceDashboard",
]
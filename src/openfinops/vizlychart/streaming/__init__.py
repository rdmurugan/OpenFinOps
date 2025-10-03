"""
Real-Time Streaming Foundation for Vizly
========================================

High-performance real-time data streaming and visualization capabilities.
Supports live data feeds, WebSocket connections, and streaming chart updates.

Features:
- WebSocket-based real-time data streaming
- Buffered data management with configurable window sizes
- Real-time chart updates with minimal latency
- Multi-client broadcasting support
- Data compression and efficient serialization
- Streaming analytics and live aggregations

Basic Usage:
    >>> from vizly.streaming import StreamingChart, DataStreamer
    >>> streamer = DataStreamer("ws://localhost:8765")
    >>> chart = StreamingChart.line()
    >>> chart.connect_stream(streamer, buffer_size=1000)
    >>> chart.start_streaming()
"""

from .data_streamer import DataStreamer, StreamingBuffer, StreamingDataManager
from .streaming_charts import StreamingChart, RealtimeLineChart, RealtimeScatterChart
from .websocket_server import WebSocketStreamer, StreamingServer
from .compression import DataCompressor, StreamingProtocol
from .analytics import StreamingAnalytics, LiveAggregator

__version__ = "0.5.0"
__all__ = [
    # Core streaming
    "DataStreamer",
    "StreamingBuffer",
    "StreamingDataManager",
    # Streaming charts
    "StreamingChart",
    "RealtimeLineChart",
    "RealtimeScatterChart",
    # WebSocket infrastructure
    "WebSocketStreamer",
    "StreamingServer",
    # Data optimization
    "DataCompressor",
    "StreamingProtocol",
    # Analytics
    "StreamingAnalytics",
    "LiveAggregator",
]
# Observability API Reference

Complete API reference for the OpenFinOps Observability Platform.

## Table of Contents

- [ObservabilityHub](#observabilityhub)
- [LLMObservabilityHub](#llmobservabilityhub)
- [CostObservatory](#costobservatory)
- [FinOpsDashboards](#finopsdashboards)
- [AIRecommendations](#airecommendations)
- [AlertingEngine](#alertingengine)

---

## ObservabilityHub

The central orchestration hub for all monitoring activities.

### Class: `ObservabilityHub`

**Location**: `openfinops.observability.observability_hub`

```python
from openfinops.observability import ObservabilityHub

hub = ObservabilityHub(config_path=None)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | str | None | Path to configuration file (YAML) |

#### Methods

##### `register_cluster(cluster_name, nodes)`

Register a compute cluster for monitoring.

```python
hub.register_cluster(
    cluster_name="gpu-cluster-1",
    nodes=["node-1", "node-2", "node-3"]
)
```

**Parameters:**
- `cluster_name` (str): Unique identifier for the cluster
- `nodes` (list[str]): List of node hostnames or IPs

**Returns:** `dict` - Registration confirmation

**Example:**
```python
result = hub.register_cluster("training-cluster", ["gpu-1", "gpu-2"])
print(result['status'])  # 'registered'
```

---

##### `get_cluster_health_summary()`

Get health summary for all registered clusters.

```python
summary = hub.get_cluster_health_summary()
```

**Returns:** `dict` - Health metrics for each cluster

**Example:**
```python
summary = hub.get_cluster_health_summary()
for cluster_name, metrics in summary.items():
    print(f"{cluster_name}: {metrics['status']}")
    print(f"  Nodes: {metrics['total_nodes']}")
    print(f"  Healthy: {metrics['healthy_nodes']}")
    print(f"  CPU Avg: {metrics['avg_cpu_usage']}%")
    print(f"  GPU Avg: {metrics['avg_gpu_usage']}%")
```

---

##### `update_node_metrics(node_id, metrics)`

Update metrics for a specific node.

```python
hub.update_node_metrics(
    node_id="gpu-node-1",
    metrics={
        "cpu_usage": 75.5,
        "gpu_usage": 92.3,
        "memory_usage": 68.2,
        "disk_usage": 45.0,
        "network_io": 1024000
    }
)
```

**Parameters:**
- `node_id` (str): Node identifier
- `metrics` (dict): Dictionary of metric values

**Returns:** `bool` - Success status

---

##### `track_cost(provider, service, cost, metadata=None)`

Track costs for cloud services.

```python
hub.track_cost(
    provider="aws",
    service="ec2",
    cost=125.50,
    metadata={
        "region": "us-west-2",
        "instance_type": "p3.2xlarge",
        "tags": {"team": "ml-research"}
    }
)
```

**Parameters:**
- `provider` (str): Cloud provider name (aws, azure, gcp, etc.)
- `service` (str): Service name
- `cost` (float): Cost amount in USD
- `metadata` (dict, optional): Additional metadata

---

## LLMObservabilityHub

Specialized monitoring for LLM training and inference workloads.

### Class: `LLMObservabilityHub`

**Location**: `openfinops.observability.llm_observability`

```python
from openfinops.observability import LLMObservabilityHub

llm_hub = LLMObservabilityHub(config_path=None)
```

#### Methods

##### `register_training_cluster(cluster_name, nodes, config=None)`

Register a training cluster with LLM-specific configuration.

```python
llm_hub.register_training_cluster(
    cluster_name="llm-training-cluster",
    nodes=["gpu-node-1", "gpu-node-2"],
    config={
        "gpu_type": "A100",
        "gpu_count": 8,
        "interconnect": "nvlink"
    }
)
```

**Parameters:**
- `cluster_name` (str): Cluster identifier
- `nodes` (list[str]): List of compute nodes
- `config` (dict, optional): Cluster configuration

---

##### `track_training_metrics(model_id, epoch, step, loss, **kwargs)`

Track training metrics for an LLM model.

```python
llm_hub.track_training_metrics(
    model_id="gpt-custom-7b",
    epoch=5,
    step=1000,
    loss=0.245,
    learning_rate=0.0001,
    gpu_memory_usage=42000,  # MB
    batch_size=32,
    gradient_norm=1.2,
    perplexity=12.5
)
```

**Parameters:**
- `model_id` (str): Unique model identifier
- `epoch` (int): Current training epoch
- `step` (int): Current training step
- `loss` (float): Training loss value
- `**kwargs`: Additional metrics (gpu_memory_usage, learning_rate, etc.)

**Returns:** `dict` - Tracking confirmation

---

##### `track_rag_metrics(system_id, query, retrieval_time, generation_time, **kwargs)`

Track RAG pipeline metrics.

```python
llm_hub.track_rag_metrics(
    system_id="production-rag",
    query="What is the capital of France?",
    retrieval_time=45.2,  # milliseconds
    generation_time=120.5,  # milliseconds
    relevance_score=0.95,
    num_retrieved_docs=5,
    total_tokens=150,
    cost=0.0025
)
```

**Parameters:**
- `system_id` (str): RAG system identifier
- `query` (str): User query
- `retrieval_time` (float): Time for document retrieval (ms)
- `generation_time` (float): Time for response generation (ms)
- `**kwargs`: Additional metrics

---

##### `track_inference_metrics(model_id, endpoint, latency, tokens, cost)`

Track inference endpoint metrics.

```python
llm_hub.track_inference_metrics(
    model_id="gpt-4",
    endpoint="production-api",
    latency=250.5,  # milliseconds
    tokens={"input": 100, "output": 250},
    cost=0.015
)
```

**Parameters:**
- `model_id` (str): Model identifier
- `endpoint` (str): Endpoint name
- `latency` (float): Response latency in ms
- `tokens` (dict): Token counts
- `cost` (float): Request cost in USD

---

##### `get_training_summary(model_id, time_range=None)`

Get training summary for a model.

```python
summary = llm_hub.get_training_summary(
    model_id="gpt-custom-7b",
    time_range="24h"  # or "7d", "30d"
)

print(f"Total steps: {summary['total_steps']}")
print(f"Current loss: {summary['current_loss']}")
print(f"Best loss: {summary['best_loss']}")
print(f"Average GPU usage: {summary['avg_gpu_usage']}%")
print(f"Total cost: ${summary['total_cost']:.2f}")
```

---

## CostObservatory

Centralized cost tracking and analysis.

### Class: `CostObservatory`

**Location**: `openfinops.observability.cost_observatory`

```python
from openfinops.observability import CostObservatory

cost_obs = CostObservatory()
```

#### Methods

##### `track_cloud_cost(provider, service, cost, region=None, **metadata)`

Track cloud service costs.

```python
cost_obs.track_cloud_cost(
    provider="aws",
    service="sagemaker",
    cost=1250.75,
    region="us-east-1",
    instance_type="ml.p3.8xlarge",
    team="ml-research",
    project="llm-training"
)
```

**Parameters:**
- `provider` (str): Cloud provider (aws, azure, gcp)
- `service` (str): Service name
- `cost` (float): Cost in USD
- `region` (str, optional): Cloud region
- `**metadata`: Additional metadata for attribution

---

##### `track_api_cost(platform, model, tokens_input, tokens_output, cost)`

Track AI API costs (OpenAI, Anthropic, etc.).

```python
cost_obs.track_api_cost(
    platform="openai",
    model="gpt-4",
    tokens_input=500,
    tokens_output=1200,
    cost=0.048
)
```

**Parameters:**
- `platform` (str): API platform (openai, anthropic, etc.)
- `model` (str): Model name
- `tokens_input` (int): Input tokens
- `tokens_output` (int): Output tokens
- `cost` (float): Request cost in USD

---

##### `get_total_cost(time_range=None, filters=None)`

Get total costs with optional filtering.

```python
# Total cost for last 30 days
total = cost_obs.get_total_cost(time_range="30d")
print(f"Total: ${total:.2f}")

# Filtered by provider
aws_total = cost_obs.get_total_cost(
    time_range="30d",
    filters={"provider": "aws"}
)
print(f"AWS Total: ${aws_total:.2f}")
```

**Parameters:**
- `time_range` (str, optional): Time range (24h, 7d, 30d, 90d)
- `filters` (dict, optional): Filter criteria

**Returns:** `float` - Total cost in USD

---

##### `get_cost_breakdown_by_provider(time_range=None)`

Get cost breakdown by cloud provider.

```python
breakdown = cost_obs.get_cost_breakdown_by_provider(time_range="30d")

for provider, cost in breakdown.items():
    print(f"{provider}: ${cost:.2f}")
```

**Returns:** `dict` - Provider to cost mapping

---

##### `get_cost_breakdown_by_service(time_range=None)`

Get cost breakdown by service.

```python
services = cost_obs.get_cost_breakdown_by_service(time_range="7d")

for service, data in services.items():
    print(f"{service}:")
    print(f"  Cost: ${data['cost']:.2f}")
    print(f"  Percentage: {data['percentage']:.1f}%")
```

---

##### `get_cost_trend(time_range, granularity="daily")`

Get cost trend over time.

```python
trend = cost_obs.get_cost_trend(
    time_range="30d",
    granularity="daily"  # or "hourly", "weekly"
)

for date, cost in trend.items():
    print(f"{date}: ${cost:.2f}")
```

**Parameters:**
- `time_range` (str): Time range
- `granularity` (str): Data granularity

**Returns:** `dict` - Date to cost mapping

---

## FinOpsDashboards

Financial operations dashboards and reporting.

### Class: `LLMFinOpsDashboardCreator`

**Location**: `openfinops.observability.finops_dashboards`

```python
from openfinops.observability import LLMFinOpsDashboardCreator

finops = LLMFinOpsDashboardCreator()
```

#### Methods

##### `create_cost_dashboard(time_range="30d")`

Create comprehensive cost dashboard.

```python
dashboard = finops.create_cost_dashboard(time_range="30d")

# Dashboard contains:
# - Total costs
# - Cost by provider
# - Cost by service
# - Cost trends
# - Top cost drivers
# - Optimization opportunities
```

**Returns:** `dict` - Dashboard data structure

---

##### `generate_cost_report(time_range, format="pdf")`

Generate cost report.

```python
report = finops.generate_cost_report(
    time_range="30d",
    format="pdf"  # or "csv", "json"
)

# Save report
with open("cost_report.pdf", "wb") as f:
    f.write(report)
```

**Parameters:**
- `time_range` (str): Reporting period
- `format` (str): Output format

**Returns:** `bytes` - Report data

---

##### `get_budget_status(budget_id)`

Check budget status.

```python
status = finops.get_budget_status("monthly-ai-budget")

print(f"Budget: ${status['budget_amount']:.2f}")
print(f"Spent: ${status['spent_amount']:.2f}")
print(f"Remaining: ${status['remaining']:.2f}")
print(f"Percentage: {status['percentage_used']:.1f}%")
print(f"Status: {status['status']}")  # ok, warning, exceeded
```

---

## AIRecommendations

AI-powered cost optimization recommendations.

### Class: `AIRecommendationEngine`

**Location**: `openfinops.observability.ai_recommendations`

```python
from openfinops.observability import AIRecommendationEngine

ai_rec = AIRecommendationEngine()
```

#### Methods

##### `get_optimization_recommendations(scope="all")`

Get AI-generated optimization recommendations.

```python
recommendations = ai_rec.get_optimization_recommendations(scope="all")

for rec in recommendations:
    print(f"Title: {rec['title']}")
    print(f"Potential Savings: ${rec['estimated_savings']:.2f}/month")
    print(f"Confidence: {rec['confidence']}%")
    print(f"Priority: {rec['priority']}")  # high, medium, low
    print(f"Action: {rec['recommended_action']}")
    print()
```

**Parameters:**
- `scope` (str): Recommendation scope (all, compute, storage, api)

**Returns:** `list[dict]` - List of recommendations

---

##### `detect_anomalies(resource_type=None, threshold=2.0)`

Detect cost anomalies using ML.

```python
anomalies = ai_rec.detect_anomalies(
    resource_type="gpu",
    threshold=2.0  # Standard deviations
)

for anomaly in anomalies:
    print(f"Resource: {anomaly['resource_id']}")
    print(f"Anomaly Score: {anomaly['score']}")
    print(f"Expected Cost: ${anomaly['expected_cost']:.2f}")
    print(f"Actual Cost: ${anomaly['actual_cost']:.2f}")
    print(f"Deviation: {anomaly['deviation']}%")
```

---

## AlertingEngine

Intelligent alerting and notification system.

### Class: `AlertingEngine`

**Location**: `openfinops.observability.alerting_engine`

```python
from openfinops.observability import AlertingEngine

alerting = AlertingEngine()
```

#### Methods

##### `create_alert_rule(name, condition, severity, channels)`

Create a new alert rule.

```python
alerting.create_alert_rule(
    name="high_gpu_cost",
    condition="gpu_cost_hourly > 500",
    severity="critical",  # critical, warning, info
    channels=["email", "slack"]
)
```

**Parameters:**
- `name` (str): Rule identifier
- `condition` (str): Alert condition (expression)
- `severity` (str): Alert severity level
- `channels` (list[str]): Notification channels

---

##### `get_active_alerts(severity=None)`

Get currently active alerts.

```python
alerts = alerting.get_active_alerts(severity="critical")

for alert in alerts:
    print(f"Alert: {alert['name']}")
    print(f"Severity: {alert['severity']}")
    print(f"Message: {alert['message']}")
    print(f"Triggered: {alert['triggered_at']}")
```

---

## Complete Example

Here's a complete example using multiple APIs together:

```python
from openfinops import ObservabilityHub, LLMObservabilityHub
from openfinops.observability import CostObservatory, AIRecommendationEngine

# Initialize components
hub = ObservabilityHub()
llm_hub = LLMObservabilityHub()
cost_obs = CostObservatory()
ai_rec = AIRecommendationEngine()

# Register cluster
llm_hub.register_training_cluster(
    cluster_name="production-cluster",
    nodes=["gpu-1", "gpu-2", "gpu-3", "gpu-4"]
)

# Track training metrics
for step in range(1, 1001):
    llm_hub.track_training_metrics(
        model_id="llm-v2",
        epoch=1,
        step=step,
        loss=1.0 / step,
        gpu_memory_usage=40000,
        batch_size=32
    )

# Track costs
cost_obs.track_cloud_cost(
    provider="aws",
    service="ec2",
    cost=450.00,
    instance_type="p3.8xlarge"
)

# Get recommendations
recommendations = ai_rec.get_optimization_recommendations()
for rec in recommendations[:3]:  # Top 3
    print(f"💡 {rec['title']}")
    print(f"   Save ${rec['estimated_savings']:.2f}/month")

# Get cost summary
total_cost = cost_obs.get_total_cost(time_range="30d")
print(f"\n💰 Total cost (30 days): ${total_cost:.2f}")
```

## See Also

- [VizlyChart API](vizlychart-api.md) - Visualization library
- [Dashboard API](dashboard-api.md) - Dashboard components
- [Telemetry Agents](telemetry-agents.md) - Cloud agents
- [Tutorials](../tutorials/basic-usage.md) - Step-by-step guides

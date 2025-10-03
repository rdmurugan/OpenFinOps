# Telemetry Agents API Reference

Complete API reference for OpenFinOps multi-cloud telemetry agents.

## Overview

OpenFinOps provides telemetry agents for collecting metrics from multiple cloud providers:
- **AWS Telemetry Agent**: Amazon Web Services metrics
- **Azure Telemetry Agent**: Microsoft Azure metrics
- **GCP Telemetry Agent**: Google Cloud Platform metrics
- **Generic Telemetry Agent**: Custom data sources

## Table of Contents

- [AWS Telemetry Agent](#aws-telemetry-agent)
- [Azure Telemetry Agent](#azure-telemetry-agent)
- [GCP Telemetry Agent](#gcp-telemetry-agent)
- [Generic Telemetry Agent](#generic-telemetry-agent)
- [Agent Configuration](#agent-configuration)
- [Deployment](#deployment)

---

## AWS Telemetry Agent

Collect metrics from AWS services.

### Class: `AWSTelemeetryAgent`

**Location**: `agents.aws_telemetry_agent`

```python
from agents.aws_telemetry_agent import AWSTelemetryAgent

agent = AWSTelemetryAgent(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    region="us-west-2"
)
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `aws_access_key_id` | str | Yes | AWS access key ID |
| `aws_secret_access_key` | str | Yes | AWS secret access key |
| `region` | str | No | AWS region (default: us-east-1) |
| `session_token` | str | No | Temporary session token |

#### Methods

##### `collect_ec2_metrics(instance_ids=None)`

Collect EC2 instance metrics.

```python
# All EC2 instances
metrics = agent.collect_ec2_metrics()

# Specific instances
metrics = agent.collect_ec2_metrics(
    instance_ids=['i-1234567890abcdef0', 'i-0987654321fedcba0']
)

for instance in metrics:
    print(f"Instance: {instance['instance_id']}")
    print(f"  CPU: {instance['cpu_utilization']:.1f}%")
    print(f"  Memory: {instance['memory_utilization']:.1f}%")
    print(f"  Network In: {instance['network_in']} bytes")
    print(f"  Network Out: {instance['network_out']} bytes")
    print(f"  Cost (hourly): ${instance['cost_per_hour']:.4f}")
```

**Returns:** `list[dict]` - List of instance metrics

---

##### `collect_sagemaker_metrics(endpoint_name=None)`

Collect SageMaker endpoint metrics.

```python
# All endpoints
metrics = agent.collect_sagemaker_metrics()

# Specific endpoint
metrics = agent.collect_sagemaker_metrics(endpoint_name="my-model-endpoint")

for endpoint in metrics:
    print(f"Endpoint: {endpoint['name']}")
    print(f"  Invocations: {endpoint['invocations']}")
    print(f"  Model Latency: {endpoint['model_latency_ms']:.2f}ms")
    print(f"  4xx Errors: {endpoint['4xx_errors']}")
    print(f"  5xx Errors: {endpoint['5xx_errors']}")
    print(f"  Cost (hourly): ${endpoint['cost_per_hour']:.4f}")
```

---

##### `collect_s3_metrics(bucket_names=None)`

Collect S3 storage metrics.

```python
metrics = agent.collect_s3_metrics(bucket_names=["my-training-data", "my-models"])

for bucket in metrics:
    print(f"Bucket: {bucket['name']}")
    print(f"  Size: {bucket['size_gb']:.2f} GB")
    print(f"  Objects: {bucket['object_count']}")
    print(f"  Storage Class: {bucket['storage_class']}")
    print(f"  Monthly Cost: ${bucket['monthly_cost']:.2f}")
```

---

##### `get_cost_and_usage(start_date, end_date, granularity='DAILY')`

Get AWS cost and usage data.

```python
import datetime

start = datetime.date(2025, 9, 1)
end = datetime.date(2025, 9, 30)

costs = agent.get_cost_and_usage(
    start_date=start,
    end_date=end,
    granularity='DAILY'  # or 'MONTHLY', 'HOURLY'
)

for day, cost_data in costs.items():
    print(f"{day}: ${cost_data['total']:.2f}")
```

---

##### `start_collection(interval_seconds=300)`

Start continuous metric collection.

```python
# Collect metrics every 5 minutes
agent.start_collection(interval_seconds=300)

# Agent runs in background
# Stop with agent.stop_collection()
```

---

## Azure Telemetry Agent

Collect metrics from Azure services.

### Class: `AzureTelemetryAgent`

**Location**: `agents.azure_telemetry_agent`

```python
from agents.azure_telemetry_agent import AzureTelemetryAgent

agent = AzureTelemetryAgent(
    subscription_id="YOUR_SUBSCRIPTION_ID",
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    tenant_id="YOUR_TENANT_ID"
)
```

#### Methods

##### `collect_vm_metrics(resource_group=None, vm_names=None)`

Collect Virtual Machine metrics.

```python
metrics = agent.collect_vm_metrics(resource_group="ml-resources")

for vm in metrics:
    print(f"VM: {vm['name']}")
    print(f"  CPU: {vm['cpu_utilization']:.1f}%")
    print(f"  Memory: {vm['memory_utilization']:.1f}%")
    print(f"  Disk IOPS: {vm['disk_iops']}")
    print(f"  Cost (hourly): ${vm['cost_per_hour']:.4f}")
```

---

##### `collect_ml_workspace_metrics(workspace_name=None)`

Collect Azure ML workspace metrics.

```python
metrics = agent.collect_ml_workspace_metrics(workspace_name="my-ml-workspace")

for workspace in metrics:
    print(f"Workspace: {workspace['name']}")
    print(f"  Active Jobs: {workspace['active_jobs']}")
    print(f"  Completed Jobs: {workspace['completed_jobs']}")
    print(f"  Failed Jobs: {workspace['failed_jobs']}")
    print(f"  Compute Hours: {workspace['compute_hours']}")
    print(f"  Monthly Cost: ${workspace['monthly_cost']:.2f}")
```

---

##### `get_cost_management_data(start_date, end_date)`

Get Azure cost management data.

```python
costs = agent.get_cost_management_data(start_date, end_date)

for service, cost in costs['by_service'].items():
    print(f"{service}: ${cost:.2f}")
```

---

## GCP Telemetry Agent

Collect metrics from Google Cloud Platform.

### Class: `GCPTelemetryAgent`

**Location**: `agents.gcp_telemetry_agent`

```python
from agents.gcp_telemetry_agent import GCPTelemetryAgent

agent = GCPTelemetryAgent(
    project_id="your-project-id",
    credentials_file="path/to/credentials.json"
)
```

#### Methods

##### `collect_compute_metrics(instance_names=None, zone=None)`

Collect Compute Engine metrics.

```python
metrics = agent.collect_compute_metrics(zone="us-central1-a")

for instance in metrics:
    print(f"Instance: {instance['name']}")
    print(f"  CPU: {instance['cpu_utilization']:.1f}%")
    print(f"  Memory: {instance['memory_utilization']:.1f}%")
    print(f"  GPU: {instance['gpu_utilization']:.1f}%")
    print(f"  Cost (hourly): ${instance['cost_per_hour']:.4f}")
```

---

##### `collect_vertex_ai_metrics(endpoint_id=None)`

Collect Vertex AI endpoint metrics.

```python
metrics = agent.collect_vertex_ai_metrics()

for endpoint in metrics:
    print(f"Endpoint: {endpoint['name']}")
    print(f"  Predictions: {endpoint['prediction_count']}")
    print(f"  Latency (p50): {endpoint['latency_p50_ms']:.2f}ms")
    print(f"  Latency (p95): {endpoint['latency_p95_ms']:.2f}ms")
    print(f"  Error Rate: {endpoint['error_rate']:.2f}%")
```

---

##### `get_billing_data(start_date, end_date)`

Get GCP billing data.

```python
billing = agent.get_billing_data(start_date, end_date)

print(f"Total Cost: ${billing['total']:.2f}")
for service, cost in billing['by_service'].items():
    print(f"  {service}: ${cost:.2f}")
```

---

## Generic Telemetry Agent

Collect metrics from custom sources.

### Class: `GenericTelemetryAgent`

**Location**: `agents.generic_telemetry_agent`

```python
from agents.generic_telemetry_agent import GenericTelemetryAgent

agent = GenericTelemetryAgent(
    endpoint="https://your-metrics-api.com",
    api_key="your-api-key"
)
```

#### Methods

##### `register_metric_collector(name, collector_func)`

Register a custom metric collector function.

```python
def collect_custom_metrics():
    # Your custom logic
    return {
        'metric1': 123.45,
        'metric2': 678.90,
        'timestamp': datetime.now()
    }

agent.register_metric_collector('custom_source', collect_custom_metrics)
```

---

##### `collect_metrics(source_name=None)`

Collect metrics from registered sources.

```python
# All sources
metrics = agent.collect_metrics()

# Specific source
metrics = agent.collect_metrics(source_name='custom_source')
```

---

##### `push_metrics(metrics, destination='openfinops')`

Push collected metrics to destination.

```python
metrics = {
    'gpu_utilization': 85.5,
    'training_loss': 0.245,
    'cost_per_hour': 3.50
}

agent.push_metrics(metrics, destination='openfinops')
```

---

## Agent Configuration

### Configuration File

Create a configuration file for agents: `config/telemetry.yaml`

```yaml
# AWS Configuration
aws:
  enabled: true
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  region: us-west-2
  services:
    - ec2
    - sagemaker
    - s3
  collection_interval: 300  # seconds

# Azure Configuration
azure:
  enabled: true
  subscription_id: ${AZURE_SUBSCRIPTION_ID}
  client_id: ${AZURE_CLIENT_ID}
  client_secret: ${AZURE_CLIENT_SECRET}
  tenant_id: ${AZURE_TENANT_ID}
  services:
    - virtual_machines
    - ml_workspace
  collection_interval: 300

# GCP Configuration
gcp:
  enabled: true
  project_id: ${GCP_PROJECT_ID}
  credentials_file: /path/to/credentials.json
  services:
    - compute
    - vertex_ai
  collection_interval: 300

# OpenFinOps Endpoint
openfinops:
  endpoint: http://localhost:8080/api/v1/metrics
  api_key: ${OPENFINOPS_API_KEY}
```

### Load Configuration

```python
import yaml
from agents.aws_telemetry_agent import AWSTelemetryAgent

# Load config
with open('config/telemetry.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize agent
agent = AWSTelemetryAgent(
    aws_access_key_id=config['aws']['access_key_id'],
    aws_secret_access_key=config['aws']['secret_access_key'],
    region=config['aws']['region']
)
```

---

## Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ ./agents/
COPY config/ ./config/

CMD ["python", "-m", "agents.aws_telemetry_agent"]
```

Build and run:

```bash
docker build -t openfinops-aws-agent .
docker run -d --name aws-agent \
  -e AWS_ACCESS_KEY_ID=xxx \
  -e AWS_SECRET_ACCESS_KEY=yyy \
  openfinops-aws-agent
```

### Kubernetes Deployment

Create `k8s/telemetry-agents.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aws-telemetry-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aws-telemetry-agent
  template:
    metadata:
      labels:
        app: aws-telemetry-agent
    spec:
      containers:
      - name: agent
        image: openfinops/aws-agent:latest
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-access-key
        - name: COLLECTION_INTERVAL
          value: "300"
```

Deploy:

```bash
kubectl apply -f k8s/telemetry-agents.yaml
```

### Systemd Service

Create `/etc/systemd/system/openfinops-agent.service`:

```ini
[Unit]
Description=OpenFinOps Telemetry Agent
After=network.target

[Service]
Type=simple
User=openfinops
WorkingDirectory=/opt/openfinops
Environment="AWS_ACCESS_KEY_ID=xxx"
Environment="AWS_SECRET_ACCESS_KEY=yyy"
ExecStart=/usr/bin/python3 -m agents.aws_telemetry_agent
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable openfinops-agent
sudo systemctl start openfinops-agent
sudo systemctl status openfinops-agent
```

---

## Complete Example

### Multi-Cloud Monitoring

```python
from agents.aws_telemetry_agent import AWSTelemetryAgent
from agents.azure_telemetry_agent import AzureTelemetryAgent
from agents.gcp_telemetry_agent import GCPTelemetryAgent
from openfinops.observability import CostObservatory
import schedule
import time

# Initialize agents
aws_agent = AWSTelemetryAgent(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region='us-west-2'
)

azure_agent = AzureTelemetryAgent(
    subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
    client_id=os.getenv('AZURE_CLIENT_ID'),
    client_secret=os.getenv('AZURE_CLIENT_SECRET'),
    tenant_id=os.getenv('AZURE_TENANT_ID')
)

gcp_agent = GCPTelemetryAgent(
    project_id=os.getenv('GCP_PROJECT_ID'),
    credentials_file='gcp-credentials.json'
)

# Initialize cost observatory
cost_obs = CostObservatory()

def collect_all_metrics():
    """Collect metrics from all cloud providers"""

    # AWS
    aws_ec2 = aws_agent.collect_ec2_metrics()
    for instance in aws_ec2:
        cost_obs.track_cloud_cost(
            provider='aws',
            service='ec2',
            cost=instance['cost_per_hour'],
            region='us-west-2',
            instance_id=instance['instance_id']
        )

    # Azure
    azure_vms = azure_agent.collect_vm_metrics()
    for vm in azure_vms:
        cost_obs.track_cloud_cost(
            provider='azure',
            service='vm',
            cost=vm['cost_per_hour'],
            vm_name=vm['name']
        )

    # GCP
    gcp_compute = gcp_agent.collect_compute_metrics()
    for instance in gcp_compute:
        cost_obs.track_cloud_cost(
            provider='gcp',
            service='compute',
            cost=instance['cost_per_hour'],
            instance_name=instance['name']
        )

    print(f"✓ Collected metrics at {datetime.now()}")

# Schedule collection every 5 minutes
schedule.every(5).minutes.do(collect_all_metrics)

print("Starting multi-cloud telemetry collection...")
while True:
    schedule.run_pending()
    time.sleep(1)
```

## See Also

- [Observability API](observability-api.md) - Data collection platform
- [Deployment Guide](../TELEMETRY_AGENT_DEPLOYMENT.md) - Detailed deployment
- [Configuration Guide](../guides/cloud-integration.md) - Cloud setup
- [Troubleshooting](../guides/troubleshooting.md) - Common issues

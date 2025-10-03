# Quick Start Guide

Get started with OpenFinOps in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/rdmurugan/OpenFinOps.git
cd OpenFinOps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Option 2: Install with Specific Features

```bash
# Install with AWS support
pip install -e ".[aws]"

# Install with all cloud providers
pip install -e ".[aws,azure,gcp]"

# Install with everything
pip install -e ".[all]"
```

## Your First OpenFinOps Program

Create a file called `my_first_monitor.py`:

```python
from openfinops import ObservabilityHub, LLMObservabilityHub

# Initialize the observability hub
hub = ObservabilityHub()
print("✓ ObservabilityHub initialized")

# Initialize LLM monitoring
llm_hub = LLMObservabilityHub()
print("✓ LLM monitoring initialized")

# Register a training cluster
llm_hub.register_training_cluster(
    cluster_name="my-gpu-cluster",
    nodes=["gpu-node-1", "gpu-node-2"]
)
print("✓ Training cluster registered")

# Track some training metrics
llm_hub.track_training_metrics(
    model_id="my-llm-model",
    epoch=1,
    step=100,
    loss=0.5,
    learning_rate=0.001,
    gpu_memory_usage=8000,
    batch_size=32
)
print("✓ Training metrics tracked")

print("\n🎉 Success! You've successfully set up OpenFinOps monitoring!")
```

Run it:

```bash
python my_first_monitor.py
```

## Run the Quick Start Example

OpenFinOps includes a quick start example:

```bash
python examples/quickstart.py
```

Output:
```
============================================================
OpenFinOps Quick Start
============================================================

1. Initializing ObservabilityHub...
   ✓ ObservabilityHub initialized

2. Initializing LLMObservabilityHub...
   ✓ LLMObservabilityHub initialized

3. Registering training cluster...
   ✓ Training cluster 'gpu-cluster-demo' registered

4. Tracking training metrics...
   ✓ Epoch 1 metrics tracked
   ✓ Epoch 2 metrics tracked
   ✓ Epoch 3 metrics tracked

============================================================
Quick Start Complete!
============================================================
```

## Basic VizlyChart Usage

Create visualizations with the built-in VizlyChart library:

```python
from openfinops.vizlychart import LineChart, ScatterChart
from openfinops.vizlychart.charts import BarChart
import numpy as np

# Create a line chart
x = np.linspace(0, 10, 100)
y = np.sin(x)

line_chart = LineChart()
line_chart.plot(x, y, color='blue', linewidth=2)
line_chart.set_title("Cost Trend Over Time")
line_chart.set_labels("Time (hours)", "Cost ($)")
line_chart.save("cost_trend.png")
print("✓ Line chart saved as cost_trend.png")

# Create a scatter chart
scatter_chart = ScatterChart()
scatter_chart.plot(
    np.random.randn(100),
    np.random.randn(100),
    color='red',
    alpha=0.6
)
scatter_chart.set_title("Resource Utilization")
scatter_chart.save("utilization.png")
print("✓ Scatter chart saved as utilization.png")
```

## Track Multi-Cloud Costs

```python
from openfinops.observability import CostObservatory

# Initialize cost observatory
cost_obs = CostObservatory()

# Track AWS costs
cost_obs.track_cloud_cost(
    provider="aws",
    service="ec2",
    cost=125.50,
    region="us-west-2",
    timestamp="2025-10-02"
)

# Track Azure costs
cost_obs.track_cloud_cost(
    provider="azure",
    service="virtual-machines",
    cost=98.75,
    region="westus",
    timestamp="2025-10-02"
)

# Get total costs
total = cost_obs.get_total_cost()
print(f"Total cloud cost: ${total:.2f}")

# Get breakdown by provider
breakdown = cost_obs.get_cost_breakdown_by_provider()
for provider, cost in breakdown.items():
    print(f"{provider}: ${cost:.2f}")
```

## Next Steps

Now that you have OpenFinOps running, explore more features:

1. **[Core Concepts](concepts.md)** - Understand the architecture
2. **[API Reference](../api/observability-api.md)** - Detailed API documentation
3. **[Tutorials](../tutorials/basic-usage.md)** - Step-by-step guides
4. **[Examples](../../examples/README.md)** - More code examples

## Common Commands

```bash
# Run the OpenFinOps CLI
openfinops --help

# Start the dashboard server
openfinops-dashboard --port 8080

# Initialize configuration
openfinops init --config config.yaml
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall in development mode
pip install -e .
```

### Module Not Found

```bash
# Check your PYTHONPATH
echo $PYTHONPATH

# Or add the src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Getting Help

- **Documentation**: [Full documentation](../README.md)
- **Examples**: Check the `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/rdmurugan/OpenFinOps/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rdmurugan/OpenFinOps/discussions)

## What's Next?

- 📊 [Build Custom Dashboards](../tutorials/custom-dashboards.md)
- 🌐 [Connect to Cloud Providers](../guides/cloud-integration.md)
- 🤖 [Monitor LLM Training](../tutorials/llm-monitoring.md)
- 💰 [Set Up Cost Attribution](../tutorials/cost-attribution.md)

Happy monitoring! 🎉

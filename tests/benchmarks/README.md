# NeMo RL Performance Benchmarks

This directory contains performance benchmarks for NeMo RL to measure and track:

- **Training throughput** (tokens/second, samples/second)
- **Memory usage** (peak, average) across backends
- **Performance regressions** compared to baselines

## Quick Start

### Run All Benchmarks

```bash
# Run all benchmarks with pytest
pytest tests/benchmarks/ -v --benchmark-enable

# Run with detailed output
pytest tests/benchmarks/ -v --benchmark-enable --benchmark-verbose
```

### Run Specific Benchmarks

```bash
# Throughput benchmarks only
pytest tests/benchmarks/test_throughput.py -v

# Memory benchmarks only  
pytest tests/benchmarks/test_memory.py -v
```

### Save Results

```bash
# Save to JSON
pytest tests/benchmarks/ -v --benchmark-enable --benchmark-json=results.json

# Compare to baseline
pytest tests/benchmarks/ -v --benchmark-enable --benchmark-compare
```

## Benchmark Structure

```
tests/benchmarks/
├── __init__.py          # Package initialization
├── utils.py             # Common utilities (config, results, baselines)
├── throughput.py        # Throughput benchmark implementations
├── memory.py            # Memory benchmark implementations
├── baselines.json       # Baseline values for regression detection
├── test_throughput.py   # Pytest throughput tests
├── test_memory.py       # Pytest memory tests
└── README.md            # This file
```

## Baselines

The `baselines.json` file contains reference performance numbers. Benchmarks compare against these baselines to detect regressions.

### Baseline Format

```json
{
  "benchmark_name": {
    "name": "benchmark_name",
    "throughput_tokens_per_sec": 10000.0,
    "peak_memory_mb": 2048.0,
    "tolerance": 0.05,
    "version": "1.0",
    "timestamp": "2026-01-15T00:00:00"
  }
}
```

### Updating Baselines

When performance improves, update baselines:

```python
from tests.benchmarks.utils import BaselineEntry, save_baseline

entry = BaselineEntry(
    name="grpo_throughput_dtensor",
    throughput_tokens_per_sec=12000.0,
    peak_memory_mb=1900.0,
    tolerance=0.05,
)
save_baseline(entry)
```

## Regression Detection

Benchmarks automatically compare results to baselines:

- **Throughput**: Regression if current < baseline × (1 - tolerance)
- **Memory**: Regression if current > baseline × (1 + tolerance)

Default tolerance is 5% (`tolerance=0.05`).

### Example Regression Check

```python
from tests.benchmarks.utils import compare_to_baseline, load_baseline

result = run_benchmark()
baseline = load_baseline(result.name)
comparison = compare_to_baseline(result, baseline)

if not comparison.passed:
    print(f"REGRESSION: {comparison.message}")
```

## CI Integration

### GitHub Actions

Add to `.github/workflows/benchmarks.yml`:

```yaml
name: Performance Benchmarks

on:
  pull_request:
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: self-hosted  # GPU runner required
    steps:
      - uses: actions/checkout@v4
      
      - name: Run benchmarks
        run: |
          pytest tests/benchmarks/ -v \
            --benchmark-enable \
            --benchmark-json=results.json
      
      - name: Check for regressions
        run: |
          python -c "
          import json
          with open('results.json') as f:
              results = json.load(f)
          # Check for failures
          if any(b.get('failed') for b in results.get('benchmarks', [])):
              exit(1)
          "
```

## Writing New Benchmarks

### Throughput Benchmark

```python
from tests.benchmarks.throughput import ThroughputBenchmark
from tests.benchmarks.utils import compare_to_baseline

def test_my_throughput():
    benchmark = ThroughputBenchmark(
        name="my_throughput_test",
        model_name="gpt2",
        batch_size=4,
        num_steps=10,
    )
    result = benchmark.run()
    
    # Compare to baseline
    comparison = compare_to_baseline(result)
    assert comparison.passed, comparison.message
```

### Memory Benchmark

```python
from tests.benchmarks.memory import MemoryBenchmark
from tests.benchmarks.utils import compare_to_baseline

def test_my_memory():
    benchmark = MemoryBenchmark(
        name="my_memory_test",
        batch_size=8,
        seq_length=1024,
    )
    result = benchmark.run()
    
    # Compare to baseline
    comparison = compare_to_baseline(result)
    assert comparison.passed, comparison.message
```

## Configuration

### BenchmarkConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `name` | Required | Unique benchmark identifier |
| `model_name` | `"gpt2"` | Model for benchmarking |
| `batch_size` | `4` | Training batch size |
| `seq_length` | `512` | Sequence length |
| `num_steps` | `10` | Number of steps to measure |
| `warmup_steps` | `2` | Warmup steps (excluded) |
| `backend` | `"dtensor"` | Training backend |
| `device` | `"cuda"` | Device to run on |

## Best Practices

1. **Run on dedicated hardware**: Avoid noisy environments
2. **Use warmup**: Exclude first few steps from measurements
3. **Consistent configuration**: Match baseline configs exactly
4. **Regular updates**: Update baselines when making optimizations
5. **Document changes**: Note any config changes in baselines

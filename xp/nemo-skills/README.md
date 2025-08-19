# NeMo-Skills Evaluation Scripts

This directory contains evaluation scripts for NeMo-Skills framework.

## GSM8K Evaluation with LLaDA (`eval_llada.py`)

A comprehensive GSM8K evaluation script that connects to your LLaDA model running as an OpenAI-compatible API server. Evaluates the full GSM8K dataset (1,319 problems) with majority voting for robust accuracy measurement.

### Prerequisites

1. **LLaDA Server**: Make sure your LLaDA server is running on localhost:8000
   ```bash
   # Test if server is accessible
   curl http://localhost:8000/v1/models
   ```

2. **NeMo-Skills Environment**: Ensure you have the NeMo-Skills package installed and available

### Usage

#### Comprehensive Evaluation (Default)
```bash
# Full GSM8K evaluation: 1,319 problems × 4 samples = 5,276 generations (~1-2 hours)
python eval_llada.py
```

#### Quick Test Mode
```bash
# Quick test: 50 problems × 1 sample = 50 generations (~5 minutes)
EVAL_QUICK_TEST=1 python eval_llada.py
```

#### Custom Configuration
```bash
# Override server URL and output directory
LLADA_SERVER_URL="http://localhost:8000/v1" EVAL_OUTPUT_DIR="./my_results" python eval_llada.py
```

### Configuration Options

The script uses the following default configuration:

```python
config = {
    "benchmarks": "gsm8k:4",  # GSM8K with 4 samples for majority voting
    "output_dir": "./eval_results",  # Results directory
    "expname": "llada-gsm8k-eval",  # Experiment name
    "server_type": "openai",  # OpenAI-compatible server
    "server_address": "http://localhost:8000/v1",  # Your server endpoint
    "model": "llada-8b-instruct",  # Model name (must match your server)
    "cluster": None,  # Local execution (no SLURM)
    "dry_run": False,  # Set to True for testing
}
```

### Environment Variables

- `LLADA_SERVER_URL`: Override the server endpoint (default: http://localhost:8000/v1)
- `EVAL_OUTPUT_DIR`: Override output directory (default: ./eval_results)
- `EVAL_QUICK_TEST`: Set to "1" for quick 50-problem test mode

### Output

The script will create evaluation results in the specified output directory:
- `eval_results/eval-results/gsm8k/metrics.json`: Final evaluation scores
- `eval_results/eval-results/gsm8k/output-rs*.jsonl`: Detailed generation outputs for each random seed
- Additional analysis and summary files

### Evaluation Modes

- **Comprehensive (Default)**: `"gsm8k:4"` - 4 samples per problem for majority voting (most robust)
- **Quick Test**: `EVAL_QUICK_TEST=1` - 50 problems with single sample (development/testing)
- **Greedy**: `"gsm8k:0"` - Single greedy sample (fastest, less robust)  
- **Robust**: `"gsm8k:8"` - 8 samples for maximum robustness (slowest)

### Troubleshooting

1. **Server Connection Issues**:
   ```bash
   # Test server connectivity
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "llada-8b-instruct", "messages": [{"role": "user", "content": "Test"}]}'
   ```

2. **Import Errors**: Make sure NeMo-Skills is properly installed and in your Python path

3. **Permission Issues**: Make the script executable: `chmod +x eval_llada.py`

4. **Memory Issues**: LLaDA evaluation can be memory intensive. Monitor system resources.

### Key Features

✅ **OpenAI API Compatible**: Works with any OpenAI-compatible server  
✅ **GSM8K Optimized**: Uses proper GSM8K prompts and few-shot examples  
✅ **Configurable**: Easy to customize via environment variables or code  
✅ **Error Handling**: Comprehensive error messages and troubleshooting tips  
✅ **Local Execution**: No cluster setup required  
✅ **Results Tracking**: Organized output with metrics and detailed logs  

### Example Output

```
============================================================
GSM8K Evaluation with LLaDA Model
============================================================
Server: http://localhost:8000
Model: llada-8b-instruct
Benchmark: gsm8k:4
Output: ./eval_results
Experiment: llada-gsm8k-eval
============================================================

[Evaluation progress...]

============================================================
EVALUATION COMPLETED SUCCESSFULLY!
Results saved to: ./eval_results
Check ./eval_results/llada-gsm8k-eval/metrics.json for final scores
============================================================
```

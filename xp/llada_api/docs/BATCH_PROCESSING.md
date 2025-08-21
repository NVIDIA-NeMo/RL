# Batch Processing for LLaDA OpenAI Server

This document explains how to enable batch processing for your LLaDA OpenAI-compatible server to improve throughput when handling multiple requests from NeMo-Skills evaluations.

## 🔍 **Current Limitation**

Your current setup processes requests **sequentially**:
```
NeMo-Skills → HTTP Request 1 → Server → Fast-dLLM → Response 1
NeMo-Skills → HTTP Request 2 → Server → Fast-dLLM → Response 2  
NeMo-Skills → HTTP Request 3 → Server → Fast-dLLM → Response 3
```

## ✅ **Solution: Batch Processing**

With batch processing, multiple requests are accumulated and processed together:
```
NeMo-Skills → HTTP Requests 1,2,3 → Server → Fast-dLLM Batch → Responses 1,2,3
```

## 🚀 **Implementation Options**

### Option 1: New Batch Server (Recommended)

Use the new `llada_batch_server.py` which provides:

- **Automatic batching**: Accumulates incoming requests
- **Smart batching logic**: Processes when batch is full OR after timeout
- **Fast-dLLM batch processing**: Uses native batch capabilities  
- **Configurable parameters**: Batch size and wait time

```bash
# Start the batch server
python llada_batch_server.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --batch-size 8 \
  --max-wait-time 0.1 \
  --port 8000
```

**Key Parameters:**
- `--batch-size 8`: Process up to 8 requests together
- `--max-wait-time 0.1`: Wait max 100ms for batch to fill
- Compatible with all your existing NeMo-Skills scripts!

### Option 2: Modify Existing Server

Use `batch_modifications.py` to add batching to your current server:

```python
# Add to llada_openai_server.py
from batch_modifications import SimpleBatchProcessor

# Initialize
batch_processor = SimpleBatchProcessor(max_batch_size=4)

# Modify endpoint
@app.post("/v1/chat/completions") 
async def create_chat_completion(request: ChatCompletionRequest):
    return await batch_processor.process_request(request)
```

## 📊 **Expected Performance Improvements**

Based on Fast-dLLM's batch capabilities:

| Batch Size | Performance | Use Case |
|------------|-------------|----------|
| 1 (current) | Baseline | Current sequential processing |
| 4 | Improved | Good balance for most cases |  
| 8 | Enhanced | High throughput evaluation |
| 16+ | Maximum | Maximum throughput (if memory allows) |

## 🛠 **Fast-dLLM Batch Support Verification**

The Fast-dLLM functions **DO support batching**:

```python
# All these functions accept batch inputs:
generate(model, batch_prompts, ...)              # Shape: (batch_size, seq_len)
generate_with_prefix_cache(model, batch_prompts, ...)
generate_with_dual_cache(model, batch_prompts, ...)

# Evidence from eval_llada.py:
batch_size = 32                    # Native batch processing
batched_input_ids.shape[0]        # Batch dimension handling
tokenizer.batch_decode()          # Batch decoding support
```

## 🔧 **Configuration for NeMo-Skills**

Your existing eval scripts work without changes:

```bash
# These work exactly the same, but faster with batching:
python eval_llada.py --quick-test
python eval_llada.py --benchmark gsm8k:4 --max-samples 100

# The server automatically batches the incoming requests
```

## ⚙️ **Tuning Batch Parameters**

### Batch Size Selection:

```bash
# Small batch: Lower latency, less throughput
python llada_batch_server.py --batch-size 2 --max-wait-time 0.05

# Medium batch: Balanced (recommended)
python llada_batch_server.py --batch-size 8 --max-wait-time 0.1

# Large batch: Maximum throughput
python llada_batch_server.py --batch-size 16 --max-wait-time 0.2
```

### Memory Considerations:

- **Larger batch sizes**: Require additional GPU memory
- Monitor GPU memory usage and adjust accordingly

## 🔍 **Monitoring & Debugging**

### Check batch stats:
```bash
curl http://localhost:8000/batch/stats
```

### Health check:
```bash
curl http://localhost:8000/health
```

### Server logs show batch processing:
```
INFO: Processing batch of 8 requests
INFO: Batch of 8 completed successfully
INFO: Using Fast-dLLM dual cache batch generation for batch size 8
```

## 🐛 **Troubleshooting**

### Issue: Requests still slow
- **Solution**: Check batch size in server logs, increase `--batch-size`

### Issue: High latency  
- **Solution**: Decrease `--max-wait-time` or use smaller batches

### Issue: Out of memory
- **Solution**: Reduce `--batch-size` or use gradient checkpointing

### Issue: Inconsistent responses
- **Solution**: Ensure all requests have similar generation parameters

## 📈 **Before/After Comparison**

### Before (Sequential):
```bash
time python eval_llada.py --quick-test --max-samples 32
# Sequential processing baseline
```

### After (Batch):
```bash  
# Start batch server
python llada_batch_server.py --batch-size 8 --max-wait-time 0.1

# Run same evaluation
time python eval_llada.py --quick-test --max-samples 32
# Improved performance with batch processing
```

## 🎯 **Recommended Setup**

For most use cases, start with:

```bash
python llada_batch_server.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --batch-size 8 \
  --max-wait-time 0.1 \
  --port 8000
```

This provides an excellent balance of throughput and latency for NeMo-Skills evaluations.

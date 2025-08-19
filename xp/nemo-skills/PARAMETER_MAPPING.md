# Parameter Mapping Between eval_llada.py and llada_openai_server.py

## ✅ **ELEGANT SOLUTION: Parameter Flow via NeMo-Skills extra_body**

This document explains how parameters flow from `eval_llada.py` through NeMo-Skills to `llada_openai_server.py`.

## 🔄 **Parameter Mapping Flow**

```
eval_llada.py → NeMo-Skills extra_body → OpenAI API request → llada_openai_server.py
```

## 📊 **Standard Parameters (Handled by NeMo-Skills)**

| eval_llada.py | NeMo-Skills Args | OpenAI API | llada_openai_server.py | Status |
|---------------|------------------|------------|------------------------|--------|
| `temperature` | `++inference.temperature` | `temperature` | `request.temperature` | ✅ Works |
| `top_p` | `++inference.top_p` | `top_p` | `request.top_p` | ✅ Works |
| `top_k` | `++inference.top_k=-1` | N/A (filtered) | `request.top_k` | ⚠️ Forced to -1 |
| `tokens_to_generate` | `++inference.tokens_to_generate` | `max_tokens` | `request.max_tokens` | ✅ Works |

### Key Insights:
- ✅ **`tokens_to_generate` → `max_tokens` mapping works perfectly** (NeMo-Skills handles this automatically)
- ⚠️ **`top_k` must be -1** (NeMo-Skills rejects other values for OpenAI compatibility)
- ✅ **`temperature` and `top_p` map correctly**

## 🎛️ **LLaDA-Specific Parameters (Via NeMo-Skills extra_body)**

**SOLUTION**: NeMo-Skills supports `extra_body` which passes custom parameters through the OpenAI API:

| eval_llada.py | NeMo-Skills Args | OpenAI API | llada_openai_server.py | Status |
|---------------|------------------|------------|------------------------|--------|
| `steps` | `++inference.extra_body.steps` | `request.steps` (via extra fields) | `request.steps` | ✅ Works |
| `block_length` | `++inference.extra_body.block_length` | `request.block_length` | `request.block_length` | ✅ Works |
| `cfg_scale` | `++inference.extra_body.cfg_scale` | `request.cfg_scale` | `request.cfg_scale` | ✅ Works |
| `remasking` | `++inference.extra_body.remasking` | `request.remasking` | `request.remasking` | ✅ Works |

### extra_body Flow:
1. `eval_llada.py` specifies `++inference.extra_body.steps=64`
2. NeMo-Skills includes this in the `extra_body` field of the OpenAI API request
3. `llada_openai_server.py` receives it as an additional field in the request body
4. Pydantic model accepts extra fields and maps them to the appropriate parameters

## 🔍 **How to Verify Parameters Are Working**

### 1. Check Server Logs
The server now logs all parameters including any received via extra_body:
```
INFO: Generation request received:
INFO:   Model: llada-8b-instruct
INFO:   Temperature: 0.7
INFO:   Max tokens: 512
INFO:   Top-p: 0.95 (NOTE: Not used in LLaDA diffusion generation)
INFO:   Top-k: -1 (NOTE: Not used in LLaDA diffusion generation)
INFO:   LLaDA steps: 128
INFO:   Block length: 64
INFO:   CFG scale: 2.0
INFO:   Remasking: random
INFO:   Extra parameters received: {'steps': 128, 'cfg_scale': 2.0, 'remasking': 'random'}
```

### 2. Test Different Parameters
```bash
# Test standard parameters
python eval_llada.py --quick-test --temperature 1.2 --tokens-to-generate 256

# Test LLaDA-specific parameters  
python eval_llada.py --quick-test --steps 128 --cfg-scale 2.0 --remasking random

# All parameters are passed via NeMo-Skills extra_body mechanism
```

### 3. Manual API Test
```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Test"}],
    "temperature": 0.8,        # ✅ Standard OpenAI parameter
    "max_tokens": 100,         # ✅ Standard OpenAI parameter (maps from tokens_to_generate)
    "top_p": 0.9,             # ✅ Standard OpenAI parameter  
    "steps": 32,              # ✅ LLaDA-specific parameter (directly in request body)
    "cfg_scale": 1.5,         # ✅ LLaDA-specific parameter
    "remasking": "random"     # ✅ LLaDA-specific parameter
})
```

## ⚠️ **Important Limitations**

1. **`top_k` is not used**: LLaDA uses Gumbel noise instead of top-k sampling
2. **`top_p` is not used**: LLaDA uses confidence-based remasking instead of nucleus sampling
3. **NeMo-Skills requires `top_k=-1`**: For OpenAI API compatibility

## 🎯 **Summary**

✅ **ELEGANT SOLUTION**: Using NeMo-Skills `extra_body` mechanism
✅ **CROSS-MACHINE COMPATIBLE**: Works when server runs on different machine
✅ **CLEAN PARAMETER FLOW**: No environment variables needed
✅ **FULLY TRANSPARENT**: Comprehensive logging shows all parameter sources

The parameter mapping now works elegantly across machines via the OpenAI API extra_body mechanism! 🎉

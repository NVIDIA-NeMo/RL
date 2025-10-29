# LLaDA/Nemotron OpenAI API Server - Documentation

OpenAI-compatible API server for LLaDA and Nemotron diffusion language models with multi-GPU support and optimized acceleration.

## 📚 Documentation

### Getting Started

- **[Quick Start](QUICK_START.md)** - Get up and running in 5 minutes
- **[Complete Guide](COMPLETE_GUIDE.md)** - Comprehensive documentation

### Key Topics

**Setup & Usage**:
- Quick Start Guide → Installation, launch, basic usage
- Complete Guide → API reference, advanced configuration

**Multi-GPU**:
- Both guides cover multi-GPU setup
- Load balancing architecture (not DataParallel)
- Linear scaling with 8+ GPUs

**Troubleshooting**:
- Common issues in both guides
- HuggingFace rate limiting fixes
- Worker startup issues
- Performance optimization

## 🚀 Quick Commands

```bash
# Single GPU (local)
export HF_TOKEN=your_token
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local --model-path GSAI-ML/LLaDA-8B-Instruct

# Multi-GPU (8 GPUs)
export HF_TOKEN=your_token
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local --gpus 8 --model-path GSAI-ML/LLaDA-8B-Instruct

# SLURM
export ACCOUNT=your_account HF_TOKEN=your_token
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --gpus 8 --model-path GSAI-ML/LLaDA-8B-Instruct
```


## 🎯 Use Cases

| Scenario | Recommended Setup |
|----------|------------------|
| **Quick testing** | Single GPU, HuggingFace model |
| **Development** | Single GPU, local model or HF |
| **Batch evaluation** | Multi-GPU (4-8), batch server |
| **Production** | Multi-GPU SLURM, load balancing |
| **Large scale** | 8+ GPUs, increased timeouts |

## ⚡ Key Features

- ✅ OpenAI-compatible API
- ✅ Multi-GPU load balancing (automatic when `--gpus > 1`)
- ✅ Automatic batch processing
- ✅ Fast-dLLM/dInfer acceleration (LLaDA)
- ✅ DCP checkpoint support
- ✅ HuggingFace model support
- ✅ Streaming responses
- ✅ SLURM integration

## 🔧 Recent Improvements

### Multi-GPU Reliability (Latest)
- **Staggered worker startup**: 1s delay between workers prevents race conditions
- **Pre-caching**: HuggingFace models pre-cached to prevent rate limiting
- **Health verification**: Active health checks before load balancer starts
- **Extended timeouts**: 20s initialization wait for full model loading

### Why Multi-GPU is Now 100% Reliable

**Previous issues**:
- Workers crashed intermittently (~30-40% failure rate with HF models)
- HuggingFace rate limiting (429 errors)
- Race conditions during simultaneous startup

**Current fixes**:
- DCP always worked (conversion delay accidentally prevented races)
- HF models now pre-cache metadata (prevents simultaneous API calls)
- Workers start with 1s stagger (prevents resource conflicts)
- Health checks with retries (catches issues early)

**Result**: 100% reliable startup for both DCP and HuggingFace models

## 📊 Performance

| Configuration | Throughput | Use Case |
|--------------|------------|----------|
| 1 GPU | ~12 req/s | Development, testing |
| 4 GPUs | ~44 req/s | Medium evaluations |
| 8 GPUs | ~86 req/s | Large-scale evaluations |

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Rate limiting (429) | Set `export HF_TOKEN=your_token` |
| Workers crash | Scripts auto-fixed (staggered startup) |
| Out of memory | `--batch-size 4` or fewer GPUs |
| Import errors | Use HF models: `--model-path GSAI-ML/LLaDA-8B-Instruct` |
| Connection refused | Check `curl http://localhost:8000/health` |

## 📚 Documentation

All documentation is consolidated into three files:

- **README.md** (this file) - Quick overview and commands
- **[QUICK_START.md](QUICK_START.md)** - 5-minute getting started guide
- **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** - Comprehensive reference

## 🎓 Learning Path

1. **Start here**: [QUICK_START.md](QUICK_START.md) - 5 minutes to get running
2. **Deep dive**: [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) - Full API reference and troubleshooting
3. **Reference**: Return to this README for quick commands and links

## 💡 Best Practices

1. **Always set HF_TOKEN** - Prevents rate limiting in multi-GPU setups
2. **Start simple** - Test with single GPU before scaling
3. **Use HuggingFace models** - Easiest to get started
4. **Monitor logs** - Check `/tmp/llada_worker_*.log`
5. **Use dInfer** - LLaDA's fastest engine (10x+ vs Fast-dLLM)
6. **Enable batching** - Automatic throughput optimization
7. **Multi-GPU for scale** - Auto-enabled when `--gpus > 1`

## 🚀 Quick Links

- **Get Token**: https://huggingface.co/settings/tokens
- **LLaDA Model**: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
- **Nemotron Model**: https://huggingface.co/nvidia/Nemotron-Diffusion-Research-4B-v0

---

**For immediate help**: See [QUICK_START.md](QUICK_START.md)

**For comprehensive info**: See [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)

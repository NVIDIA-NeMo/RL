```
uv cache clean
rm -rf .venv venvs
time uv run nemo_rl/utils/prefetch_venvs.py 'Vllm|DTensor'
```

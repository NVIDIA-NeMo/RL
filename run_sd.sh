#!/bin/bash
uv run python examples/run_self_distillation.py 2>&1 | tee run_self_distillation_$(date +%Y%m%d_%H%M%S).log

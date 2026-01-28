# Fault Tolerance Launcher Guide

The `ft_launcher` is provided by `nvidia-resiliency-ext` (included in NeMo RL dependencies) and enables automatic fault tolerance and recovery for distributed training runs.

## Key Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--ft-cfg-path` | Path to FT YAML config file | `examples/configs/ft_config.yaml` |
| `--ft-rank-heartbeat-timeout` | Heartbeat timeout in seconds | `450` |
| `--ft-initial-rank-heartbeat-timeout` | Initial timeout (longer for setup) | `1200` |
| `--max-restarts` | Maximum number of restart attempts | `5` |

## Basic Usage

```bash
uv run ft_launcher \
    --ft-cfg-path examples/configs/ft_config.yaml \
    --ft-rank-heartbeat-timeout 450 \
    --ft-initial-rank-heartbeat-timeout 1200 \
    --max-restarts 5 \
    examples/run_grpo_math.py \
    --config <your_config.yaml>
```

## FT Config File (examples/configs/ft_config.yaml)

```yaml
fault_tolerance:
  initial_rank_heartbeat_timeout: 360
  restart_policy: any-failed
```

## Important Notes

1. **Checkpointing**: Enable checkpointing for recovery to work:
   ```bash
   ++checkpointing.enabled=true
   ++checkpointing.checkpoint_dir=/path/to/checkpoints
   ++checkpointing.save_period=50
   ```

2. **Timeouts**: Set `--ft-initial-rank-heartbeat-timeout` higher than `--ft-rank-heartbeat-timeout` to allow for model loading/setup time.

3. **Restart Policy**: The `any-failed` restart policy will restart the entire job if any rank fails. Look for these log messages to identify when a restart occurs:

   ```
   [ERROR] [ft_launcher...] failed (exitcode: 1) local_rank: 0 (pid: ...) of binary: ...
   [INFO] [ft_launcher...] [default] Worker group FAILED. 3/5 attempts left; will restart worker group
   [INFO] [ft_launcher...] Stopping workers... Timeout = 30 sec.
   [INFO] [ft_launcher...] The node '...' attempts to join the next round of the rendezvous '...'.
   [INFO] [ft_launcher...] The node '...' has joined round N of the rendezvous '...' as rank 0 in a world of size 1.
   ```

   Key indicators:
   - `Worker group FAILED. X/Y attempts left` - shows a restart is happening and remaining attempts
   - `will restart worker group` - confirms restart is in progress
   - `has joined round N` - the round number increases with each restart

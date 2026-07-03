# Remote Sparse-Delta vLLM Refit
For non-colocated Megatron policy workers and sync vLLM workers that share the
same checkpoint. Policy workers keep a CPU baseline and stream zstd-compressed
sparse deltas through either S3 or ZeroMQ. Both transports share export,
encoding, compression, backpressure, receiver apply, and transactional baseline
commit logic.
Payload checksums and transfer-scoped IDs make HTTP retries idempotent. Policy
workers commit their baselines only after every receiver flush succeeds.
On a fresh run, generation starts from the shared checkpoint while policy
workers build the CPU baseline asynchronously; the first transfer follows the
first optimizer step. Resumed runs synchronize before generation.

## Config

```yaml
backend: vllm
colocated: {enabled: false}
refit_transport: vllm_s3_sparse
delta_compression:
  dtype: bf16
  sparse_bucket_size_bytes: 268435456
vllm_cfg:
  async_engine: false
  http_refit_server_port: 8081
  http_refit_api_key_env_var: NRL_REFIT_API_KEY
```

Use `refit_transport: vllm_zmq_sparse` and set
`vllm_cfg.zmq_refit_server_port` when Kubernetes needs a stable ZeroMQ target
port. The ZeroMQ service must route TCP traffic from policy workers to the vLLM
relay workers; each relay fans a payload out to every HTTP refit endpoint. On a
flat cluster network, the dynamically reported worker IP can be used directly;
a service mesh is not required.

Remote sparse refit requires `kv_cache_dtype: auto`; FP8 KV-cache scale sync is not
supported. Receiver tensors must have a direct QKV, MoE, Mamba, or generic TP
placement plan; transformed and FP8 weights fail before any delta is applied.
The HTTP endpoints and ZeroMQ producer relay use
`http_refit_api_key_env_var` auth when configured.

For S3, set `NRL_REFIT_S3_BUCKET` and, when needed,
`NRL_REFIT_S3_REGION` or `NRL_REFIT_S3_PREFIX`. AWS CRT performs multipart
transfer automatically. Tune export, encode, and transfer concurrency with
`NRL_REFIT_S3_EXPORT_CHUNK_BYTES`, `NRL_REFIT_S3_ENCODE_WORKERS`, and
`NRL_REFIT_S3_UPLOAD_WORKERS`.

For ZeroMQ, tune the same stages with `NRL_REFIT_ZMQ_EXPORT_CHUNK_BYTES`,
`NRL_REFIT_ZMQ_ENCODE_WORKERS`, and `NRL_REFIT_ZMQ_SEND_WORKERS`. Relay
concurrency is controlled by `NRL_REFIT_ZMQ_RELAY_PAYLOAD_WORKERS` and
`NRL_REFIT_ZMQ_RELAY_FANOUT_WORKERS`; fanout defaults to 32 workers to preserve
HTTP keep-alive reuse and avoid receiver contention. Track `REFIT_S3_TIMING` or
`REFIT_ZMQ_TIMING`, `REFIT_RECEIVER_TIMING`, and `REFIT_*_GLOBAL_COMMIT` in
cluster runs.

Set `NRL_REFIT_VERIFY_SAMPLES_PER_PAYLOAD` to a small positive value to verify
deterministic transmitted-delta samples after placement. Each receiver snapshots
only those target elements before apply and compares `post - pre` with the
placement- and dtype-adjusted transmitted delta, so an existing absolute weight
offset cannot contaminate the metric. `REFIT_*_DELTA_VERIFY` reports candidate
and applied sample counts, exact mismatches, tolerance-gated mismatches
(`rtol=1e-6`, `atol=1e-8`), and mean/max absolute delta error. A gated mismatch
aborts the transaction before baseline commit. Successful commits advance the
CPU baseline to the quantized value applied by the receiver, so later deltas
compensate compression residuals instead of accumulating drift.

`REFIT_*_DELTA_CHANGE` reports changed and total exported element counts plus
their model-wide percentage. The codec accumulates these counters while it is
already finding sparse locations; it does not perform another tensor scan.
When a training logger is enabled, the same values are emitted to W&B and
TensorBoard under `refit/delta/*`, `refit/delta_verify/*`, and
`refit/transfer/*`. End-to-end refit latency remains available as
`timing/train/prepare_for_generation/transfer_and_update_weights`.

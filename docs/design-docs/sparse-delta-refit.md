# Remote Sparse-Delta vLLM Refit
For non-colocated Megatron policy workers and sync vLLM workers that share the
same checkpoint. Policy workers keep a CPU baseline and stream zstd-compressed
sparse deltas through either S3 or ZeroMQ. Both transports share export,
encoding, compression, backpressure, receiver apply, and transactional baseline
commit logic.

## Config

```yaml
backend: vllm
colocated: {enabled: false}
refit_transport: vllm_s3_sparse
delta_compression:
  enabled: true
  dtype: bf16
  sparse_bucket_size_bytes: 268435456
vllm_cfg:
  async_engine: false
  expose_http_refit_server: true
  http_refit_server_port: 8081
  http_refit_api_key_env_var: NRL_REFIT_API_KEY
```

Use `refit_transport: vllm_zmq_sparse` and set
`vllm_cfg.zmq_refit_server_port` when Kubernetes needs a stable ZeroMQ target
port. The ZeroMQ service must route TCP traffic from policy workers to the vLLM
relay workers; each relay fans a payload out to every HTTP refit endpoint.

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
`NRL_REFIT_ZMQ_RELAY_FANOUT_WORKERS`. Track `REFIT_S3_TIMING` or
`REFIT_ZMQ_TIMING`, `REFIT_RECEIVER_TIMING`, and `REFIT_*_GLOBAL_COMMIT` in
cluster runs.

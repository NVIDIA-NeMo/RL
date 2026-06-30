# S3 Sparse-Delta vLLM Refit
For non-colocated Megatron policy workers and sync vLLM workers that share the
same checkpoint. Policy workers keep a CPU baseline, upload zstd-compressed
sparse deltas to S3, post receiver manifests, and commit the baseline only after
the global flush succeeds.

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

S3 sparse refit requires `kv_cache_dtype: auto`; FP8 KV-cache scale sync is not
supported. Receiver tensors must have a direct QKV, MoE, Mamba, or generic TP
placement plan; transformed and FP8 weights fail before any delta is applied.
The receiver exposes `/nemo-rl/refit/s3-manifest`, with
`http_refit_api_key_env_var` auth when configured. Export chunks are capped by
`NRL_REFIT_S3_EXPORT_CHUNK_BYTES` and
`delta_compression.sparse_bucket_size_bytes`. Set `NRL_REFIT_S3_BUCKET` and,
when needed, `NRL_REFIT_S3_REGION` or `NRL_REFIT_S3_PREFIX`. AWS CRT performs
multipart transfer automatically; encode and end-to-end pipeline concurrency
default from available CPU cores and can be fixed with
`NRL_REFIT_S3_ENCODE_WORKERS` and `NRL_REFIT_S3_UPLOAD_WORKERS`. Track `REFIT_S3_TIMING`,
`REFIT_RECEIVER_TIMING`, and `REFIT_S3_GLOBAL_FLUSH` in cluster runs.

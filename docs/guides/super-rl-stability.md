# Super RL stability changes on PR 3941

This branch starts at `ca06137460b7e2edcaf6f1fd67ddbda5ddd6b8e2`.
Changes are extracted from the September 12–14 Super-s25 failure ledger,
not a copy of its run directory. Each fix includes its scope and validation
boundary. No job IDs, user paths, credentials, datasets, or checkpoints are
required by the utilities. Historical validation is not validation of this
refactored branch on every cluster.

Required workloads are **CCC, CoT, TIR/ns_tools, SciCode, and equivalence
judging**. Removing a route, disabling reasoning, lowering the agreed output
budget, or converting a service failure into reward zero is not a repair.

## CCC metadata staging: no driver dependency on orjson

**Incident 1:** a preparation driver lacked `orjson`; staging exited before
writing verifier data. Moving the job to Gym Python worked historically but
was unnecessary for a JSON-only transformation.

`tools/super_rl/stage_ccc.py` uses only stdlib `json`. It streams source
JSONLs, verifies source SHA-256 and each authorized problem's hash, retains
native tests/graders/subtasks, and atomically publishes without overwriting.
It does not edit training prompts, routes, budgets, or SciCode data.

Supply a manifest with this shape (use real independently verified hashes):

```json
{
  "sources": [{"path": "chunk.jsonl", "sha256": "<source-byte-sha256>"}],
  "problems": [{"competition_id": "c", "problem_id": "p", "sha256": "<problem-sha256>"}]
}
```

Problem hashing uses sorted-key compact UTF-8 JSON. A mismatch with a historical
serializer **fails**; do not regenerate authority hashes just to make it pass.
Paths resolve relative to the manifest. Create the output parent first, then
run in a CPU allocation using the preparation environment:

```bash
uv run tools/super_rl/stage_ccc.py --manifest /data/ccc-manifest.json --output /run/data/ccc.jsonl
```

Validation: `tests/unit/tools/test_stage_ccc.py` covers exact problem content,
hash failures, missing/duplicate identities, strict JSON, and no-clobber output.
Real CCC resources-server loading and sandbox execution remain native smoke
gates; a JSON test does not certify verifier throughput.

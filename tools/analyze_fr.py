"""Analyze NCCL flight-recorder dumps (job 5355697, step-0 stall) to find the
stuck collective per rank and separate culprits (behind) from victims (waiting)."""
import glob
import os
import pickle
from collections import Counter, defaultdict

files = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), "fr_*")),
    key=lambda p: int(p.rsplit("_", 1)[1]),
)
print(f"loaded {len(files)} rank dumps")

# rank -> stuck collective (the oldest non-completed 'started', else first 'scheduled')
stuck = {}          # rank -> (pg_name, seq, op, in_sizes)
trailing = {}       # rank -> count of scheduled backlog
pg_members = {}     # pg_name -> set of global ranks that ever ran on it

for rank, f in enumerate(files):
    d = pickle.load(open(f, "rb"))
    entries = d.get("entries", [])
    started_open = None
    first_sched = None
    n_sched = 0
    for e in entries:
        st = e.get("state")
        pg = e.get("process_group", ("?", "?"))
        pgname = pg[1] if isinstance(pg, (list, tuple)) and len(pg) > 1 else str(pg)
        if st == "started" and not e.get("time_discovered_completed_ns"):
            # newest open 'started' wins (the actual in-flight block)
            started_open = (pgname, e.get("collective_seq_id"),
                            e.get("profiling_name"), tuple(e.get("input_sizes") or []))
        elif st == "scheduled":
            n_sched += 1
            if first_sched is None:
                first_sched = (pgname, e.get("collective_seq_id"),
                               e.get("profiling_name"), tuple(e.get("input_sizes") or []))
    stuck[rank] = started_open or first_sched
    trailing[rank] = n_sched

# aggregate: which (pg, op) are ranks blocked in
by_pg_op = Counter()
by_pg_seq = defaultdict(list)   # (pgname, op) -> list of (rank, seq)
for rank, s in stuck.items():
    if s is None:
        by_pg_op[("<none>", "<all-completed>")] += 1
        continue
    pgname, seq, op, insz = s
    by_pg_op[(pgname, op)] += 1
    by_pg_seq[(pgname, op)].append((rank, seq))

print("\n=== ranks blocked, grouped by (process_group, collective) ===")
for (pg, op), n in by_pg_op.most_common():
    seqs = by_pg_seq.get((pg, op), [])
    seqvals = sorted({s for _, s in seqs})
    print(f"  {n:3d} ranks  pg={pg:32s} op={op:28s} seq(s)={seqvals[:6]}"
          + (" ..." if len(seqvals) > 6 else ""))

# culprit detection: within each pg+op, if ranks are split across seq ids,
# the ones at the LOWER seq are behind (culprits); higher seq are waiting (victims).
print("\n=== seq split within each blocked (pg, op) -> culprit(lower) vs victim(higher) ===")
for (pg, op), rankseqs in by_pg_seq.items():
    seqcount = Counter(s for _, s in rankseqs)
    if len(seqcount) > 1:
        lo = min(seqcount)
        hi = max(seqcount)
        lo_ranks = sorted(r for r, s in rankseqs if s == lo)
        print(f"  pg={pg} op={op}: {len(seqcount)} distinct seqs")
        print(f"     LOWEST seq={lo} ({seqcount[lo]} ranks, BEHIND/culprit): {lo_ranks[:16]}")
        print(f"     HIGHEST seq={hi} ({seqcount[hi]} ranks, waiting/victim)")

# trailing backlog: ranks with the biggest scheduled backlog are furthest behind
print("\n=== ranks with largest 'scheduled' backlog (most behind) ===")
for rank, n in sorted(trailing.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  rank {rank:3d}: {n} scheduled queued behind its block; stuck={stuck[rank]}")

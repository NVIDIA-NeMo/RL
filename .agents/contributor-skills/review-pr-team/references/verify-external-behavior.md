# Verify external behavior

Claims about how Python, a library, a renderer or upstream code behaves must be checked.

- **Run a 10-line repro for any "Python/the runtime will lose/mask/wait/reorder" claim and quote the output.** Four lanes agreed `drain()` was the fix on #3592; a repro showed `asyncio.run` joins the executor anyway.
- **Read the pinned upstream source before downgrading a finding that turns on one yes/no fact.** On #3591 four raw fetches of NCCL at the pinned tag turned a ~57% note into the top finding.
- **Never rest a finding or a fix on a WebFetch summary of code; download the raw file and `wc -l` it.** On #3798 every cited line number was past the end of a 2030-line file, and the summary said the opposite of the code.
- **For "the new version drops the old plugin", read the new loader's accepted-version table.** NCCL 2.30 still loads net plugin v6–v12 (#3352); a vendor "tested with" matrix proves nothing.
- **Render markdown with `gh api markdown` before calling it broken.** GFM joined a URL split across lines on #3688; CommonMark rules are not GitHub's renderer.
- **Before calling a train/inference asymmetry sloppy, grep the upstream validator for the field.** Megatron-LM requires `moe_router_dtype == "fp32"` for `inference_optimized` (#3490).
- **"Upstream permits it" earns a wording fix, not "delete this line from a validated recipe".** The author had run the config on GB200; nobody had run the version without the line (#2279).
- **Check every step of the author's failure chain for the exact SKU or config you cite, and mark each step verified or "author's claim".** FlashAttention 2 allows D>192 on GB200 but not GB300, so the proposed GB200 evidence run would show nothing (#4439).
- **Quote the error the user sees last, not the first `raise`.** A retry wrapper printed the clear message six times and died on a generic one after ~63s (#3388).
- **If you did not run it, write the finding as the guard that is bypassed, not as an invented symptom.** Lanes offering "corrupts" vs "deadlocks" vs "shape error" is the sign nobody ran it (#3654).

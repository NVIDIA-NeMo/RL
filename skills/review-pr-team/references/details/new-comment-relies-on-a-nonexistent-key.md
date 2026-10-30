# new-comment-relies-on-a-nonexistent-key

> When a PR adds a comment that explains a nearby config key, grep the pinned upstream for the key; a comment can reach the right conclusion through a key that never existed

A new comment can be right in its conclusion and still rest on a mechanism that does not exist. If the
key it explains is a pre-existing dead key (a context line, not a `+` line), both halves pass a normal
diff scan: the key is old, and the conclusion is correct. The comment then becomes the only thing
asserting the key is real, and the next reader treats it as load-bearing.

**Why:** On #3785 two prefetch YAMLs added a comment saying fields "must be set even on thin clients
that never spin up a server", relying on a `spinup_server` key. The key did not exist in NeMo-Gym at
the pin: `git grep` and `git log -S` were both empty, and the only occurrences were the YAML lines
themselves. The fields really were required, so the conclusion held; only the reason was fiction. The
author dropped the dead key lines and reworded the comment around Gym's real gate.

**How to apply:**
1. For every new comment that explains a config key, run `git grep <key> <pin-sha>` and
   `git log -S "<key>" --all` in the submodule.
2. If both are empty, the finding is the dead key plus the comment that documents it; ask to drop the
   key and reword the comment around the real mechanism.
3. When the author rewords, re-derive every clause of the replacement rather than accepting it.

**Seen:** #3785

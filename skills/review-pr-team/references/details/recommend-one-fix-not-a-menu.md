# recommend-one-fix-not-a-menu

> Review comments state one recommended fix — no "Pick one / Preferred / If you'd rather" menus, no "optional" escape hatches

Every review comment gives the author ONE recommended fix. Do not write "Pick one.", "**Preferred:** … **If you'd rather …:** …", "one way to do it — several valid shapes", or "X is optional to change". If a pre-existing thing should be left alone, say "leave it as it is"; if a fix sketch is illustrative, just give it.

**Why:** On PR #4129 the user had me strip the choice from every staged comment. Their ask on the `generation_backend` config field was "derive it; if we must mutate, rename to `_generation_backend`" — I posted both as an ordered menu, and they wanted only the recommendation. A menu hands the decision back to the author and invites the weaker option.

**How to apply:** Before staging, grep every body (outside suggestion blocks) for `pick one|preferred|if you'd rather|if that is more|one way to do it|alternatively|is optional|are optional` and rewrite each hit as a single recommendation. If you genuinely cannot pick, that is a question for the user before posting, not a menu for the author. Related: `proposed-fixes-need-their-own-adversarial-pass` (see [proposed-fixes](../proposed-fixes.md)).

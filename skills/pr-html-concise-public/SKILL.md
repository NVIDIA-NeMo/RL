---
name: pr-html-concise-public
description: One-page-per-question HTML explainer for a PR — classify the PR first: a new feature owes a page comparing it with the existing way, a fix owes a page showing the failure and the fix, and a PR with both owes two cross-linked pages. Leads with a diagram, plain words. Uploads to the user's own PUBLIC gh-pages-poc GitHub Pages site.
argument-hint: "<pr-number> [the one thing to explain]"
user-invocable: true
---

# One-Page PR Explainer (public)

Make **one page** that tells a reader everything they need to know about one PR
(or one question inside a PR). A human reads it top to bottom in two minutes and
is done. No agent has to summarize it afterward.

**Repo**: `NVIDIA-NeMo/RL` (unless the user names another).
**Upload**: the user's own copy of the `gh-pages-poc` GitHub Pages site (see
"Site setup and upload" below) — this site is **public to the internet**.

"One page" is the default *per question*, not per PR. A PR that adds a feature
**and** has a notable bug is two questions, so it is two pages — see Step 0.

---

## The bar

Three pages set the standard. Read at least one before you write:

- https://terrykong.github.io/gh-pages-poc/terryk/pr-3427-sc-ownership.html
- https://terrykong.github.io/gh-pages-poc/terryk/pr-3427-sc-samplers.html
- https://terrykong.github.io/gh-pages-poc/terryk/pr3429-cursor-advance.html

Each is one question, one great figure, one small worked example, one clear
consequence. Each is 15–25 KB of HTML. Nothing else.

**Before you upload, be your own harsh critic.** Put your page next to those
three, blind, and ask: *which one is better?* Not "is mine acceptable" — which
one would a tired engineer rather open on their phone. If yours is not the
winner, you are not done. Go back and cut, redraw, and rewrite until a critic
who has never seen your PR looks at your page and is genuinely impressed.

The critic you have to impress cares about exactly four things:

1. **Do I know whether this is broken, fixed, or only proposed?**
2. **Did the figure teach me the thing before I read any prose?**
3. **Do I understand how the fix works — could I point at the code that does it?**
4. **Could I read it without stopping to look up a word?**

Do not soften the bar. Do not ship a page you would skim past yourself.

---

## Step 0: Classify the PR, then pick the page set

**Do this before you choose a title, and before you pick a figure.** The most
common way this skill goes wrong is not a bad page — it is a good page answering
the wrong question. The pull is always toward the most *interesting* thing you
found (usually a bug), when the reader's first question is the most *important*
one: what is this change, and how does it fit next to what already exists.

Read the diff and the PR description, then classify:

| The PR is… | Tell from | Pages you owe the reader |
|---|---|---|
| **A new feature / new path / new component** | new modules, a new config block, a new dataset/worker/estimator, a new branch threaded through several layers | **(1) what it is + how it compares to the existing way** — mandatory, and it goes first. Then (2) a page per deep finding, if any. |
| **A fix** | changes an existing behavior, no new user-facing surface | one page: the failure and the fix |
| **A refactor / migration** | same behavior, new shape | one page: what moved and what stayed the same |
| **A fix inside a new feature** | both of the above in one PR | **two pages** — the feature comparison first, the bug second, cross-linked. Do not fold the bug into the feature page, and do not ship only the bug page. |

### When it is a new feature, the first page is a comparison

A reader who has never seen the feature cannot judge a bug in it. If the PR adds
a second way to do something the codebase already does, the first page must answer:

- **What is this, in one sentence?**
- **What already existed that does the same job?** Name it and link it. If nothing
  did, say that — "there was no way to do X before" is a real answer.
- **Where does each one do the work?** Point at the function, in the driver, the
  dataset, the worker — wherever it actually happens. A figure with both paths
  stacked, packing/branching box highlighted in each, beats any prose.
- **What does the data look like at each hop, in both?** Same tiny example carried
  through both paths, with real shapes. Show it *in the call-flow figure itself*, as
  an abbreviated value next to every step (see "Show the data, not just the steps"
  in §6), not in a separate table the reader has to line up by hand.
- **Do they converge?** Almost always yes, at some layer. Name the exact function
  and line where the two become one, and say what still differs *after* that point.
  "They merge at X, except for fields A and B" is the single most useful sentence
  on the page.
- **Why keep both?** What each one is for, and what the new one gives up in exchange.

### Rule

**Never ship only a bug page for a PR that introduces a new feature.** A bug page
alone tells a reviewer to go fix line 291; it does not tell them whether the design
is right, which is the thing they actually have to approve. If you only have budget
for one page, write the comparison — the bug belongs in the review comments anyway.

Say your classification out loud to the user before you build, in one line:
"This is a new feature plus a possible major bug, so I'll make two pages." That
gives them a chance to redirect you before you spend the effort.

---

## Hard limits

These are limits, not targets. Breaking one means you picked too big a topic —
split it into two pages instead (that is what `pr-3427-sc-ownership` and
`pr-3427-sc-samplers` are: two pages for one PR).

| Limit | Value |
|---|---|
| Length | 1–2 screens of scrolling |
| `<h2>` sections | 3 maximum |
| Figures | 1 for the problem. A second for the fix, when a picture beats pseudocode. |
| File size | Aim under 25 KB |
| Table of contents | None. If you need one, the page is too long. |
| Lead paragraph | 5 sentences maximum |

---

## Structure

Write it in this order. Every part is required except where noted.

### 1. Title — the question, in plain words

An `<h1>` that states what the reader will learn, as a sentence a person would
say out loud:

> How the dataloader cursor advances past in-flight prompts at checkpoint save

Not "PR #3429 analysis". Not "SingleController checkpointing deep dive".

### 2. Subtitle — one line of provenance

`<p class="sub">`: link to the PR, link to any related issue, and the pinned
head SHA that all permalinks use. One line.

**If the page explains a review comment, link that comment.** This applies whether
the whole page is about one finding or the page gives each finding its own section.
The reader's next move is to go to the comment and act on it, so hand them the link
instead of making them hunt the Files tab for it. Put it at the end of the section
it belongs to, and label it with the file and line so it reads as a destination:

```html
<p class="cref">🔴 <a href="https://github.com/OWNER/REPO/pull/PRNUM#discussion_rNNNNNNNNNN">review comment — generic_sft.py:259</a></p>
```

```css
.cref{margin:6px 0 0;font-size:13px}
.cref a{display:inline-block;padding:4px 10px;border:1px solid var(--border);
  border-radius:16px;background:var(--surface);font:12.5px ui-monospace,Menlo,monospace}
.cref a:hover{background:var(--surface2);text-decoration:none}
```

Read the URLs out of the API rather than assembling them by hand — every review
comment already carries the right anchor in its `url` field:

```bash
gh api graphql -f query='query { repository(owner:"OWNER", name:"REPO") {
  pullRequest(number: PRNUM) { reviews(last:1) { nodes { comments(first:100) {
  nodes { url path line } } } } } } }'
```

**A pending review's comments are visible only to its author.** Those anchors do
not resolve for anyone else until the review is submitted, so say that plainly when
you hand over the page — otherwise the links look broken to the first reader who
opens it.

### 3. Status — is this broken, fixed, or proposed?

**Goes directly under the subtitle, above the link bar and the lead.** This is the single most
common thing these pages leave out, and it costs the reader the whole page: they land on a
description of a problem and cannot tell whether they are reading about a bug that still exists, a
bug this PR repairs, or a fix somebody is only suggesting. Every other sentence reads differently
depending on that answer.

Three rows, in this order, one line each:

| Row | What goes in it |
|---|---|
| **Before** | What the code did before this PR — the status quo, stated as behavior, not as code. |
| **This PR** | What the change actually does. One sentence. |
| **Status** | The verdict, in bold and in words. |

The verdict is one of:

- **Fixed by this PR** — the page describes a bug the change repairs.
- **Open — not fixed in this PR** — the page describes a problem the change introduces or leaves
  behind. Say where it is tracked (review comments, an issue), so the reader knows it is not lost.
- **Proposed — not implemented** — the page argues for a change nobody has made yet.
- **Pre-existing, untouched by this PR** — the page explains something the PR merely sits next to.

Rules:

- **Never make the reader infer the verdict from the tone.** A page full of red boxes reads like a
  live bug even when it is describing something the PR fixes. Two pages can look almost identical
  and mean opposite things; only the status box tells them apart.
- **A section that proposes a fix must say so in its heading** — "What would fix it — proposed, not
  in the PR" — and open with "Nothing below is implemented." A reader who skims a code block will
  otherwise take it for the diff.
- When the PR both fixes something and leaves something open, that is two questions and therefore
  two pages (Step 0). Each gets its own status box, and each links to the other.

```html
<div class="status">
<dl>
  <dt>Before</dt>
  <dd>Setting the flag changed nothing — the code ran, cleared no cache, and reported success.</dd>
  <dt>This PR</dt>
  <dd>Hands the flag to the engine's own pause, which really does drop the cached state.</dd>
  <dt>Status</dt>
  <dd><span class="verdict good">Fixed by this PR.</span> Both modes now behave as documented.</dd>
</dl>
</div>
```

```css
.status{border:1px solid var(--border);border-radius:10px;background:var(--surface);
  padding:2px 16px 10px;margin:0 0 20px}
.status dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:12px 0 0}
.status dt{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.04em;
  white-space:nowrap;padding-top:2px}
.status dd{margin:0;font-size:14px}
.verdict{font-weight:700}
.verdict.good{color:var(--green)}
.verdict.open{color:var(--red)}
```

### 4. Sibling pages — the link bar

One PR often gets several pages, one question each. **Every page for a PR links
to every other page for that PR**, so a reader who lands on any one of them can
find the rest. The links go both ways: the new page lists the old ones, and you
go back and add the new page to each old one.

Put the bar right after the subtitle, whenever the PR has more than one page:

```html
<nav class="chain">
  <a href="https://OWNER.github.io/gh-pages-poc/DIR/pr-3591-where-it-runs.html">where it runs</a>
  <span class="here">what survives</span>
  <a href="https://OWNER.github.io/gh-pages-poc/DIR/pr-3591-why-rebuild-not-shrink.html">why rebuild, not shrink</a>
</nav>
<p class="chainnote">Three pages on this PR, one question each — this one: which failures recover.</p>
```

The CSS, copied as-is into the `<style>` block:

```css
nav.chain{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 6px;padding:0}
nav.chain a,nav.chain span{display:inline-block;padding:5px 11px;border:1px solid var(--border);
  border-radius:20px;background:var(--surface);text-decoration:none;color:var(--accent);
  font-size:12.5px;line-height:1.3;white-space:nowrap}
nav.chain a:hover{background:var(--surface2)}
nav.chain span.here{background:var(--accent);color:var(--bg);border-color:var(--accent);font-weight:600}
p.chainnote{color:var(--dim);font-size:12px;margin:0 0 18px}
```

Rules:

- The label is 2–4 plain words naming that page's question — not its filename.
- The current page is a `<span class="here">`, not a link.
- Same order on every page. If the pages read as a sequence, number the labels
  (`1 · who watches what`); if each stands alone, leave the numbers off.
- Use full site URLs, so the bar works from anywhere.
- Skip the bar entirely when this is the only page for the PR.

### 5. Lead — the mechanism, in 3–5 sentences

What the code does, enough that the figure makes sense. Every claim about the
code carries a permalink to the exact line. No history, no motivation essay, no
"this PR is part of a larger effort".

### 6. The figure — this is the page

An inline `<svg>` that carries the whole argument. A reader who looks only at
the figure should get it. Rules:

- **Colors come from CSS variables** (`fill:var(--text)`, `var(--green)`, …) via
  classes, never hardcoded, so the light/dark toggle re-themes it live.
- **Every code reference in the figure is a clickable permalink**, wrapped in an
  SVG `<a>` and underlined (`svg a text{text-decoration:underline}`). Assume the
  reader is on a phone — there is no hover.
- **Label the states with color and a legend in the figure itself.** Green =
  fine, red = the problem, gray = ignored. Put the legend inside the SVG, next
  to what it labels, not in a caption below.
- **Annotate every arrow** with what call it is and which line does it.
- Draw the *specific* thing, not a generic box diagram. A grid of 30 numbered
  prompts colored by fate beats three boxes labeled "loader / buffer / trainer".
- **Show the data, not just the steps.** In a call-flow or two-path figure, every
  step box gets the abbreviated value that goes in and comes out, written as code:
  a short JSON/dict, a struct with its key fields, or a tensor with shape and a few
  values — `[{role:"user", content:"A: "}, …]` → `"A: Paris"` →
  `ids [32, 25, 12095]` → `turn 2: tensor([], dtype=int64)`. Use the same tiny input
  in both lanes, at the same height, so the reader can tell at a glance whether the
  two sides hold the same thing at each step and spot the one step where they split.
  A list of function names says what runs; the data says what it does. Abbreviate
  freely (`…`, `[B=2, S=7]`, the first 3 values), but every value must come from a
  real run, not be made up. Mark the step where the two lanes differ in red.
- Check the SVG tags are balanced before uploading, **and that no text runs past
  the viewBox width**. SVG does not wrap and gives no warning — a long label just
  vanishes off the right edge in the browser while looking fine in the source.
  Estimate each label as `len(text) x font-size x 0.55` (sans) or `x 0.6` (mono),
  and confirm `x + width < viewBox_width - 14`, accounting for `text-anchor`.
  Long annotations belong **under** the chart, left-aligned — never in the strip
  to the right of a marker line, where there is rarely room.

If the topic genuinely has no picture, use a tight aligned monospace block
instead — but try hard for the picture first.

### 7. The fix — show the mechanism, not just the outcome

On a fix PR the figure almost always shows the **problem**. That is half a page.
The reader must leave knowing *how* the code fixes it. Give the fix its own
section, in one of three forms:

- **A real diff — the default whenever the fix is a code change**, whether it
  is in the PR or only proposed. Show it as a color-coded unified diff, not as
  loose snippets or pseudocode: seeing which lines go and which arrive is how a
  reader gets a feel for the size and shape of the change. Rules below.
- **A second figure** — an illustration of the new mechanism, held to the same
  rules as §6. Use this when the fix is a new component, or a changed order of
  operations. It can sit above the diff.
- **Pseudocode with a value trace** — 2–4 short blocks, one per step of the fix,
  each with inline comments carrying the **actual values** at that point, and a
  permalink line underneath pointing at the real code. Use this only when there
  is no single code change to show (a protocol across processes, an ordering
  argument). The reader should be able to follow the real Python by reading
  your comments.

**Diff rules:**

- Real code, not a sketch of it: every `-` line must exist at the pinned SHA,
  and every `+` line must be what you would actually commit. Keep 1–3 unchanged
  context lines around each change, so the reader sees where it lands.
- One hunk per place the code changes. Head each hunk with the file, the line
  numbers at the pinned SHA, and the function — `@@ single_controller.py
  L4549-L4553 · _save_rollout_checkpoint_locked`. Only print line numbers you
  checked; do not estimate the new-file offsets of a `@@ -a,b +c,d @@` header.
  A brand-new function gets a hunk too: "new method, next to `X` (L1949)".
- Color from the theme variables — removed red, added green, hunk headers
  accent — so the light/dark toggle works. Put a one-line comment inside the
  `+` code wherever the new line is not obvious; the prose around the diff does
  not travel with it.
- Say under the diff what each piece does, in order, with permalinks, and say
  plainly if it is proposed and has not been run.

```html
<pre class="diff"><span class="hunk">@@ single_controller.py L4549-L4553 · _save_rollout_checkpoint_locked</span>
         if self._optimizer_commit_in_progress:
             return _RolloutCheckpointSaveResult(...)
<span class="del">-        old line
</span><span class="add">+        # why this line exists
+        new line</span></pre>
```

```css
.diff{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface);
  border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;white-space:pre;margin:10px 0}
.diff .add{color:var(--green)}
.diff .del{color:var(--red)}
.diff .hunk{color:var(--accent)}
```

**Say where the fix lands.** Open the fix section with one line naming the repo(s)
the change goes in — "RL only", "Gym only", "RL and Gym" — and the files or
functions, each linked. When a repo needs no change, say so and why in one clause
("No Gym change: Gym already accepts an ACK for any finished run"). A reader deciding
who owns the fix, or whether a second PR in another repo is needed, should not have
to work it out from the diff. Put it in a box so it cannot be missed:

```html
<p class="where"><b>Where the fix lands: RL only, two files.</b>
<a href="…/rollout_manager.py#L2408-L2442"><code>rollout_manager.py</code></a>: … .
<b>No Gym change:</b> Gym already … .</p>
```

```css
.where{border-left:4px solid var(--accent);background:var(--surface);padding:6px 12px;margin:6px 0 10px}
```

Whichever you pick, run it on the **same tiny numbers as the figure**, so §6 and
§7 are one worked example and not two. Real arithmetic: 2 prompts, not 512. Pick
numbers that make the interesting case show up, and line the state up column by
column at the moment that matters.

**Pick numbers that cannot be confused with each other.** Every quantity in the
example — each config value you set *and* every value derived from them — must be
distinct. When two of them share a number, the reader cannot tell which one a
given digit refers to, and the example teaches nothing while looking like it does.

List them before you draw, derived values included. A real case that failed this:

| quantity | value | |
|---|---|---|
| `policy_training_start_step` | 1 | |
| `max_trajectory_age_steps` | 1 | ← same as the one above |
| `warmup_max_trajectory_age_steps` | 2 | |
| frontier (derived, `= 1 + 1`) | 2 | ← same as the one above |

Four concepts, two numbers. "frontier = 1 + 1 = 2" is untraceable, and "the age
limit is 2" and "the frontier is 2" read as one fact rather than two. Re-picking
as 2 / 1 / 4 (frontier 3) makes all four distinct, and every digit on the page
traces back to exactly one source.

Primes are a useful nudge, not the rule — 2, 1 and 3 are all prime, yet `2 + 1 = 3`
collides with the third. The test is distinctness **after** you compute the derived
values. Prefer small distinct numbers; if the clean set costs one more column than
the colliding one, spend the column.

The same rule applies to counts you draw: if there are 3 workers, do not also use 3
steps or a batch of 3. Vary them so a reader can tell which 3 a label means.

### 8. So what — one callout

A `.warn` or `.hl` box with the consequence: what breaks, what gets faster, what
the reader should do. Two to four sentences. This is the last thing on the page.

---

## Formula background — when the PR changes the math

Some PRs can only be understood as a change to an equation. For those, put a short
formula block near the top of the page, before the figures. On PR #4670 (single-forward
CISPO) this block was what made the page click: the reader could see which term became 1,
which logprob was dropped, and which term still handled async lag.

**Add one when the page's argument is a claim about an equation** — a reader has to see
a term go to 1, cancel, be swapped, or change between modes before they believe it:

- The PR touches training math: a loss or objective, advantage estimation, KL, importance
  or ratio terms, clipping, normalization, or reward shaping.
- A config flag fixes, drops, or replaces a term (`force_on_policy_ratio`,
  `use_kl_in_reward`, `token_level_loss`, the TIS type).
- The page needs to say "X = 1 here", "this equals Y", or "only the logged value differs".
- Several quantities have look-alike names (`prev` / `curr` / `gen` logprobs, the
  reference policy) and the reader must keep them apart.

**Skip it** for plumbing (checkpointing, scheduling, config wiring, CI), for a PR that only
moves code without changing the math, and when the formula would be one obvious line the
prose already says.

**What the block looks like:**

1. **Name every symbol first**, each next to its code name: "π<sub>gen</sub>
   (`gen`): vLLM's probability when it sampled the token". Say where it comes from, what it
   costs, and whether gradients flow through it.
2. **One formula for how it works today.** Put each term in `\underbrace{…}_{\text{(1) …}}`
   and give it its own color. Under the formula, give one plain line per term saying what
   it measures, plus a permalink to the code that computes it.
3. **State the assumption the PR uses, in one sentence.** Example: "in the first optimizer
   update on a batch, π<sub>θ</sub> and π<sub>prev</sub> are the same weights, so (2) is 1".
4. **The formula after the PR's change**, in the same colors, so the reader sees exactly
   which term changed.
5. **A mode table when the answer depends on how the run is set up** (sync / async, one vs
   several updates per batch, …). One cell per case: which terms are 1, and whether the
   PR's mode is allowed. Then one sentence on which axis actually decides.
6. **Match the code, not the paper.** Check that the formula matches what the code really
   does: which terms are detached (write them as `\mathrm{sg}[…]`), the order the terms
   multiply in, and what *N* divides by. If it also equals the paper's formula under some
   reading, say so in one sentence.

**Mechanics:**

- Load KaTeX at a pinned version, render each `<div class="tex">` with
  `katex.render(src, el, {displayMode:true, trust:true, strict:false, throwOnError:false})`:

  ```html
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script>window.addEventListener('load',function(){if(!window.katex)return;
    document.querySelectorAll('.tex').forEach(function(e){try{katex.render(e.textContent,e,
      {displayMode:true,trust:true,strict:false,throwOnError:false});}catch(x){}});});</script>
  ```

- Color terms with `\htmlClass{t1}{…}` and CSS such as `.t1{color:var(--orange)}`, never a
  hard-coded `\color{#hex}`, so the light/dark toggle recolors the math too. Use the same
  `.t1`/`.t2`/`.t3` classes in the prose and in the table, so a term keeps one color
  everywhere on the page.
- Give `.tex` `overflow-x:auto`, because a long formula must scroll on a phone, not overflow.
- Before upload, render every formula with the CLI and check that the colored classes come
  through: `npx -y katex@0.16.9 --display-mode --trust -i f.tex -o f.html`, then
  `grep 'class="enclosing t1"' f.html`. The CLI's "strict mode" warnings about
  `\htmlClass` are expected; the page itself passes `strict:false`.
- The block is background, not a figure: it does not count toward the figure limit. Keep
  it to two formulas at most (before and after). If you need a third, the page is about
  two questions.

---

## Language

**Assume the reader arrives with 0–10% of your context.** They have not read the
PR, the diff, or the surrounding code. Write so they never have to stop and look
something up. This is not dumbing it down — it is refusing to make them pay for
your shorthand.

**The biggest failure mode: inventing shorthand.** While reading the diff you
pick up words for things — *the ring*, *the frontier*, *the stamp*, *the cursor*,
*the ordinal* — and then use them on the page as if they were common knowledge.
They are not. They are yours, from an hour ago. Every one of them must be either
(a) replaced by what the thing plainly is, or (b) named once, right where it is
explained, and then used.

| Your shorthand | What the reader can follow |
|---|---|
| "the ring keeps the last 512 snapshots" | "the last 512 loader places are kept" |
| "stamped with its ordinal" | "given a number: 0, 1, 2, …" |
| "saved at the trained frontier" | "saved at the point training had reached" |
| "the cursor is past the batch" | "the loader has already moved on" |

**Real code names are not shorthand — quote them.** `frontier_ordinal`,
`_dataloader_snapshots`, `state_dict()` are things the reader can grep for, so
keep them in the code blocks and put the plain meaning in the comment beside
them. The rule is against *prose* built out of invented nouns.

**Explain the subsystem, not just the diff.** The reader does not know the
feature, and often not the subsystem it lives in either (a trainer engineer
reading about Gym, a Gym engineer reading about Megatron). Two tools, both short:

1. **A background box before the lead**, whenever the page leans on a subsystem
   the reader may not use. 3–5 lines: its parts, what each one does, and why the
   PR needs them. No history.
2. **Tooltips** on every term, route, config value, or state name the sentence
   does not already explain. One or two sentences each: what it is and what it
   does. Permalinks still go to the code; the tooltip adds meaning without
   leaving the page. The sentence must read fine without opening it.

Never let a bare identifier stand in for an explanation. "RL asks every Gym
server `GET /ng-control/v1/capabilities`" teaches nothing. Write "RL asks every
Gym server what it can do in a checkpoint", and put the route in the tooltip.

```css
.tip{border-bottom:1px dotted var(--dim);cursor:help;position:relative;outline:none}
.tip:hover::after,.tip:focus::after{content:attr(data-tip);position:absolute;left:0;top:1.7em;z-index:60;
  width:max-content;max-width:min(340px,80vw);white-space:normal;background:var(--surface2);color:var(--text);
  border:1px solid var(--border);border-radius:6px;padding:7px 10px;font:13px/1.45 -apple-system,sans-serif;
  font-weight:400;box-shadow:0 4px 14px rgba(0,0,0,.3)}
```

```html
<span class="tip" tabindex="0" data-tip="Gym's HTTP client. It looks up a server's address by its config name.">ServerClient.request</span>
```

`tabindex="0"` makes it open on tap, since phones have no hover. In the figure,
add `<title>` to box labels for desktop hover, but the label must still stand
alone, because phones never show it. Say once, in the background box: "Dotted
words have a short explanation: hover or tap them."

**Rules:**

- Short sentences. One idea each.
- Common words only. If a word would not appear in a normal spoken conversation
  between two engineers, replace it.
- Say the thing directly. No "it is worth noting that", "importantly",
  "as we can see", "arguably".
- Active voice: "the save drops these prompts", not "these prompts are dropped".
- Define a term the first time you use it, in the same sentence. Domain words
  like *rollout*, *refit*, *cursor* are fine — one short clause is enough.
- Explain what a thing *does*, not what category it belongs to.
- The figure's labels obey this too — they are read *first*, with the least
  context of anything on the page.

**Never use these. Say the real thing instead:**

| Don't write | Write |
|---|---|
| load-bearing | "X has to happen before Y" / "other code depends on this" |
| non-trivial | big, hard, slow |
| orthogonal | unrelated |
| leverage | use |
| surface (as a verb) | show |
| semantics | what it means |
| canonical | the standard one |
| idempotent | safe to run twice |
| pathological | worst case |
| degenerate | empty, or all-one-value |
| elide / obviate / preclude | skip / make unnecessary / prevent |
| salient | important |
| delta | difference |
| amortized | spread across many steps |
| in the limit | when N gets large |
| by construction | because of how it is built |
| nuanced | there is a catch |

The list is examples, not the whole rule. The rule is: **plain, common words.**
Read your draft out loud in your head. Anywhere you would not say it that way to
a colleague at a whiteboard, rewrite it.

---

## Public site — check before you upload

This site is readable by anyone on the internet. `NVIDIA-NeMo/RL` is a public
repo so the code and PR content are fine, but the page must not contain:

- Internal cluster names, hostnames, DNS names, or cloud provider / datacenter
  locations. Say the **GPU SKU only** — "GB200", "H100".
- Colleagues' names, emails, or chat handles. Refer to "the author" / "the
  review". GitHub handles that already appear on the public PR are acceptable,
  but leave them out unless they are needed.
- Internal URLs (internal Git servers, intranet), ticket IDs, credentials, tokens,
  keys, or absolute paths that show internal infrastructure.
- Unreleased model names, internal metrics, roadmap or revenue numbers.

GitHub Pages cannot be made private: a Pages site is served publicly even from a
private repo, except on GitHub Enterprise Cloud. If you are not sure something is
cleared for public release, **stop and ask** — do not upload.

---

## Build and upload

1. **Find the pages that already exist for this PR.** Pull the site repo and
   look for them:

   ```bash
   git -C ~/gh-pages-poc pull --ff-only
   ls ~/gh-pages-poc/$DIR/pr-$PRNUM-*.html ~/gh-pages-poc/$DIR/pr$PRNUM-*.html 2>/dev/null
   ```

   (`$DIR` and the one-time site setup are in "Site setup and upload" below.)

   Open each hit and read its `<h1>` — that is where its nav label comes from.
   They go in the new page's link bar (§4), and each of them gets a link back to
   the new page in step 6.
2. Gather: `gh pr view $PRNUM --repo NVIDIA-NeMo/RL --json title,body,author,headRefOid,files`,
   the diff, and the review comments. Pin every permalink to `$HEAD_SHA`.
3. Write the HTML. Start from the structure of `pr3429-cursor-advance.html`
   (fetch it — it is small and it is the template): GitHub-dark CSS variables,
   the light/dark toggle below, `.wrap` at `max-width:980px`.
4. Self-critique against the three reference pages. Cut. Redraw. Repeat.
5. Upload (see below) as `$DIR/pr-$PRNUM-<slug>.html`, where
   `<slug>` is 2–3 kebab-case words naming the question
   (`cursor-advance`, `sc-ownership`, `sc-samplers`).
6. **Link the old pages back.** Edit every sibling from step 1: add the new page
   to its `<nav class="chain">` in the same order the new page uses, add the bar
   and its CSS if that page does not have one yet, and fix its `.chainnote`
   count. Push those edits together with the new page.
7. Poll until the page serves the new content, then give the user the URL — plus
   the URLs of the sibling pages you edited.

---

## Site setup and upload

The pages are served by GitHub Pages from a repo named `gh-pages-poc` under the
user's own GitHub account. Its workflow copies every HTML file into the site and
builds an index page sorted by last commit. The reference copy is
`terrykong/gh-pages-poc`.

### One-time setup: fork the site repo

```bash
OWNER=$(gh api user --jq .login)
if ! gh repo view "$OWNER/gh-pages-poc" >/dev/null 2>&1; then
  gh repo fork terrykong/gh-pages-poc --clone=false
  # GitHub turns Actions off on new forks, and Pages starts disabled. Turn both on.
  gh api -X PUT "repos/$OWNER/gh-pages-poc/actions/permissions" -F enabled=true -f allowed_actions=all
  gh workflow enable pages.yml --repo "$OWNER/gh-pages-poc"
  gh api -X POST "repos/$OWNER/gh-pages-poc/pages" -f build_type=workflow
  gh workflow run pages.yml --repo "$OWNER/gh-pages-poc"
fi
[ -d ~/gh-pages-poc/.git ] || gh repo clone "$OWNER/gh-pages-poc" ~/gh-pages-poc
```

Ask the user before running this the first time: it creates a public repo under
their account. If the `pages` call fails (for example, an org policy blocks Pages),
stop and tell the user to turn Pages on under the fork's Settings → Pages, with
"GitHub Actions" as the source. The fork also carries the existing example pages;
that is harmless, since they are already public.

### Where files go

`$DIR` is the part of `git config user.email` before the `@` (`jdoe@example.com` →
`jdoe`). If it is unset, or `whoami` is `root` (likely a container), ask the user.
Files go in `~/gh-pages-poc/$DIR/`, and `$DIR/foo.html` is served at
`https://$OWNER.github.io/gh-pages-poc/$DIR/foo.html`.

### Commit, push, and check it is live

```bash
cd ~/gh-pages-poc
git add "$DIR/<file>.html"          # name each file; never `git add -A`
git commit -s -m "Add <file>.html"
git pull --rebase && git push
```

GitHub Pages rebuilds asynchronously (about 30–90 s), so the URL 404s or serves the
old version for a while. First make sure the build succeeded — a failed build never
updates the page:

```bash
gh run watch "$(gh run list --repo "$OWNER/gh-pages-poc" --limit 1 --json databaseId -q '.[0].databaseId')" \
  --repo "$OWNER/gh-pages-poc" --exit-status
```

Then poll for HTTP 200 **and** a string that only the new version contains:

```bash
URL="https://$OWNER.github.io/gh-pages-poc/$DIR/<file>.html"
NEEDLE='a string only in the new version'
for i in $(seq 1 15); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL")" = 200 ] && curl -s "$URL" | grep -qF "$NEEDLE"; then
    echo "live (attempt $i)"; break
  fi
  sleep 8
done
```

Hand the user the URL only after it serves the new content.

### Light/dark toggle (every page)

Drive **every** color from CSS variables on `:root`, override them under
`:root[data-theme="light"]`, and add a fixed toggle button that flips `data-theme`
on `<html>` and saves the choice. Default is dark. SVG text set to `var(--text)`
re-themes live. Any hardcoded dark background (code blocks) needs a light override.

```css
:root{--bg:#0d1117;--surface:#161b22;--surface2:#1c2128;--border:#30363d;--text:#e6edf3;--dim:#8b949e;
  --accent:#58a6ff;--green:#3fb950;--red:#f85149;--orange:#d29922}
:root[data-theme="light"]{--bg:#ffffff;--surface:#f6f8fa;--surface2:#eaeef2;--border:#d0d7de;--text:#1f2328;
  --dim:#59636e;--accent:#0969da;--green:#1a7f37;--red:#cf222e;--orange:#9a6700}
body{background:var(--bg);color:var(--text)}
#theme-toggle{position:fixed;top:14px;right:16px;z-index:200;background:var(--surface);
  border:1px solid var(--border);color:var(--text);border-radius:20px;padding:6px 12px;
  font:13px/1 ui-monospace,monospace;cursor:pointer}
```

Just before `</head>`, apply the saved theme before first paint:

```html
<script>(function(){try{if(localStorage.getItem('gh-pages-theme')==='light')
  document.documentElement.setAttribute('data-theme','light');}catch(e){}})();</script>
```

Just after `<body>`, the button and its wiring:

```html
<button id="theme-toggle" aria-label="Toggle light/dark theme">🌙 dark</button>
<script>(function(){var r=document.documentElement,K='gh-pages-theme';
  function lbl(){var b=document.getElementById('theme-toggle');if(b)b.textContent=r.getAttribute('data-theme')==='light'?'☀ light':'🌙 dark';}
  function wire(){var b=document.getElementById('theme-toggle');if(!b)return;
    b.onclick=function(){if(r.getAttribute('data-theme')==='light'){r.removeAttribute('data-theme');try{localStorage.setItem(K,'dark')}catch(e){}}
      else{r.setAttribute('data-theme','light');try{localStorage.setItem(K,'light')}catch(e){}}lbl();};lbl();}
  document.readyState!=='loading'?wire():document.addEventListener('DOMContentLoaded',wire);})();</script>
```

Check that the inline JS parses before uploading (for example `node --check` on the
extracted script); a syntax error there blanks the whole page.

---


## Checklist

- [ ] You classified the PR (Step 0) and said the classification out loud before building
- [ ] If the PR adds a new feature, a comparison page exists and comes first —
      you did not ship only a bug page
- [ ] The comparison page names where each path does the work, carries the same
      tiny example through both, and says where they converge and what still differs
- [ ] A status box sits under the subtitle — Before / This PR / Status — with the
      verdict spelled out in words, not left to the reader's inference
- [ ] Any section proposing an unimplemented fix says so in its heading
- [ ] The fix section says where the fix lands: which repo(s), which files/functions,
      and which repo needs no change and why
- [ ] One question, answered. Not a survey of the PR.
- [ ] Every section that explains a review comment links that comment, and you said
      whether the review is still pending (its links resolve only for its author)
- [ ] If the PR has other pages, this one links to all of them, and every one
      of them was edited to link back to this one
- [ ] Fits in 1–2 screens; 3 or fewer `<h2>`; under ~25 KB
- [ ] The figure alone teaches the main point
- [ ] Every step in a call-flow or two-path figure shows the abbreviated data going
      in and out (JSON/struct/tensor from a real run), lined up across the two lanes
- [ ] Figure colors use `var(--…)`; light/dark toggle re-themes everything
- [ ] No SVG text runs past the viewBox width (nothing silently clipped)
- [ ] Every quantity in the worked example is distinct, derived values included
- [ ] Every code reference in the figure is a clickable, underlined permalink
- [ ] Legend lives inside the figure, next to what it labels
- [ ] Every claim about the code has a permalink pinned to `$HEAD_SHA`
- [ ] The **fix** is shown, not just the problem — a color-coded diff when the fix
      is a code change (real `-`/`+` lines, context, verified hunk headers), a
      second figure for a new mechanism, or pseudocode with a value trace only
      when there is no single code change to show
- [ ] The example uses tiny numbers and lines up column by column, and §6 and §7
      use the *same* numbers
- [ ] No word from the banned list; no word a non-native reader would stop on
- [ ] If the PR changes training math (a loss, a ratio, a flag that fixes or drops a term),
      a formula block names every symbol, shows before/after with colored terms, states the
      assumption, matches the code (sg, term order, N), and renders in the KaTeX CLI
- [ ] Read it as someone who knows neither the feature nor the subsystem: a
      background box exists if the page leans on a subsystem, every term the
      sentence does not explain has a 1–2 sentence tooltip, and no bare route,
      config key, or enum value stands in for an explanation
- [ ] Read it as someone with 0% context: is there a single noun (in the prose
      *or* the figure labels) that you only know because you read the diff? If
      yes, replace it or explain it where it first appears
- [ ] Passed the public-content check (no cluster names, no colleagues, GPU SKU only)
- [ ] You compared it blind against the three reference pages and yours won

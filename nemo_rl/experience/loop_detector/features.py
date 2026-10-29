# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Features of one thinking turn for the loop detector. Stdlib only, linear time in the turn length.

Masking (prose()): fenced code blocks, inline code, URLs and lines that look like code, logs, diffs, tables or data
(code-symbol or digit share of the line's non-space characters > 0.25, or indented by 4+ spaces) are removed. Numbers
inside prose lines are kept, so templated listings with changing values (test lists, arithmetic, per-field
walkthroughs) do not look repetitive.

Detector A (frozen, features_turn(text, "A")):
  a_sym_rep8  share of 8-token windows of the masked prose already seen earlier in the turn; tokens are lowercased
              words, numbers and single symbols (so "— ✗ — ✗" or "⇒⇒⇒" filler counts), punctuation dropped
  a_sym_k5    share of those windows seen at least 5 times before (loops repeat a phrase many times; echoed text
              repeats it once or twice)
  a_raw_k3    share of 8-grams of the raw, unmasked whitespace tokens seen at least 3 times before (catches
              re-quoting the same output again and again)
  a_filler    filler words per prose word: hmm/hm, "hold on", wait, ok/okay, actually, oh, ugh, argh
  a_log_len   log10 of the raw turn length in characters

Detector B (frozen, features_turn(text, "B"); compression and lexical diversity, no n-gram or filler counts):
  b_gz        zlib ratio of the masked prose (compressed / raw bytes)
  b_gz_low    share of 2,000-char prose chunks whose own zlib ratio is below 0.25
  b_ttr_mean  mean distinct-word share over consecutive 100-word windows
  b_ttr_low   share of 100-word windows with distinct-word share below 0.3
  b_punct     "!", "?" and dash characters per word
  b_caps      all-caps words of 2+ letters per word (WAIT, OK, GO, IMPOSSIBLE)
  b_log_len   log10 of the raw turn length

Development-only candidates (all_features; not used by the frozen detectors): word-only rep8 / rep8_k5, longest run
of repeated windows (run_share), last-third minus first-third repetition (tail_head), duplicate-fragment share
(frag_dup).
"""
import math
import re
import zlib

FENCE = re.compile(r"```.*?```", re.S)
TICK = re.compile(r"`[^`\n]{0,300}`")
URL = re.compile(r"https?://\S+")
CODECH = set("{}[]()<>=;:/\\|_#$@*^~+`\"'%&0123456789")
WORD = re.compile(r"[A-Za-z][A-Za-z']*")
SYMTOK = re.compile(r"[a-z0-9']+|[^\sa-z0-9.,;:()\[\]{}\"`]")
FRAG = re.compile(r"[.!?;,\n]+|\s[-—–]+\s|—|–")
FILL1 = {"hmm", "hm", "hmmm", "hmmmm", "wait", "ok", "okay", "actually", "oh", "ugh", "argh"}


def prose(text):
    t = URL.sub(" ", TICK.sub(" ", FENCE.sub("\n", text)))
    keep = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            continue
        if line.startswith("    ") or line.startswith("\t"):
            continue
        ns = len(s) - s.count(" ")
        if ns and sum(c in CODECH for c in s) / ns > 0.25:
            continue
        keep.append(s)
    return "\n".join(keep)


def raw_mult(text, n=8, k=3):
    """Share of 8-grams of raw whitespace tokens (no masking, lowercased) seen at least k times before."""
    w = text.lower().split()
    m = len(w) - n + 1
    if m <= 0:
        return 0.0
    cnt, rk = {}, 0
    for i in range(m):
        g = " ".join(w[i:i + n])
        c = cnt.get(g, 0)
        if c >= k:
            rk += 1
        cnt[g] = c + 1
    return rk / m


def _ngram_feats(words, n=8):
    m = len(words) - n + 1
    if m <= 0:
        return dict(rep8=0.0, rep8_k5=0.0, run_share=0.0, tail_head=0.0)
    cnt = {}
    rep = rep5 = run = best = 0
    a, b = m // 3, m - m // 3
    head = tail = 0
    for i in range(m):
        g = " ".join(words[i:i + n])
        c = cnt.get(g, 0)
        if c:
            rep += 1
            run += 1
            if run > best:
                best = run
            if c >= 5:
                rep5 += 1
            if i < a:
                head += 1
            elif i >= b:
                tail += 1
        else:
            run = 0
        cnt[g] = c + 1
    return dict(rep8=rep / m, rep8_k5=rep5 / m, run_share=best / m,
                tail_head=(tail / max(1, m - b)) - (head / max(1, a)))


def features_a(text, p=None):
    p = prose(text) if p is None else p
    words = [w.lower() for w in WORD.findall(p)]
    n = len(words)
    f = _ngram_feats(words)
    fs = _ngram_feats(SYMTOK.findall(p.lower()))
    f.update(sym_rep8=fs["rep8"], sym_k5=fs["rep8_k5"], raw_k3=raw_mult(text))
    low = p.lower()
    fill = sum(w in FILL1 for w in words) + low.count("hold on")
    frags = [" ".join(WORD.findall(x.lower())) for x in FRAG.split(p)]
    frags = [x for x in frags if x]
    seen, dup = set(), 0
    for x in frags:
        if x in seen:
            dup += 1
        else:
            seen.add(x)
    f.update(filler=fill / max(1, n), frag_dup=dup / max(1, len(frags)), log_len=math.log10(max(1, len(text))),
             n_words=n)
    return f


def features_b(text, p=None):
    p = prose(text) if p is None else p
    raw = p.encode("utf-8", "ignore")
    gz = len(zlib.compress(raw, 6)) / max(1, len(raw))
    chunks = [p[i:i + 2000] for i in range(0, len(p), 2000)]
    chunks = [c for c in chunks if len(c) >= 500]
    low = sum(1 for c in chunks if len(zlib.compress(c.encode("utf-8", "ignore"), 6)) / max(1, len(c.encode("utf-8", "ignore"))) < 0.25)
    words = [w.lower() for w in WORD.findall(p)]
    n = len(words)
    ttr = [len(set(words[i:i + 100])) / 100 for i in range(0, n - 99, 100)]
    punct = sum(p.count(c) for c in "!?—–")
    caps = sum(1 for w in WORD.findall(p) if len(w) >= 2 and w.isupper())
    return dict(gz=gz, gz_low=low / max(1, len(chunks)), ttr_mean=(sum(ttr) / len(ttr)) if ttr else 1.0,
                ttr_low=(sum(1 for x in ttr if x < 0.3) / len(ttr)) if ttr else 0.0, punct=punct / max(1, n),
                caps=caps / max(1, n), log_len=math.log10(max(1, len(text))), n_words=n)


def all_features(text):
    p = prose(text)
    fa = features_a(text, p)
    fb = features_b(text, p)
    out = {f"a_{k}": v for k, v in fa.items()}
    out.update({f"b_{k}": v for k, v in fb.items()})
    out["prose_share"] = len(p) / max(1, len(text))
    return out


def features_turn(text, detector="A"):
    """Only the features the frozen detector uses (detector.py); values equal those of all_features."""
    p = prose(text)
    if detector == "A":
        low = p.lower()
        fs = _ngram_feats(SYMTOK.findall(low))
        words = WORD.findall(low)
        fill = sum(w in FILL1 for w in words) + low.count("hold on")
        return {"a_sym_rep8": fs["rep8"], "a_sym_k5": fs["rep8_k5"], "a_raw_k3": raw_mult(text),
                "a_filler": fill / max(1, len(words)), "a_log_len": math.log10(max(1, len(text)))}
    return {f"b_{k}": v for k, v in features_b(text, p).items()}

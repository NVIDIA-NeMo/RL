# flag-long-in-function-comments

> Long (5+ line) comment blocks inside function bodies are a review finding (tech debt, context cost, rot); comments on declarations/config fields are fine; list every in-function block with a one-line replacement

Count the comment blocks of 5+ lines that a PR adds **inside function bodies** (find each block's enclosing scope with `ast`; don't eyeball it). If there are many, raise one comment:
- anchor it on the worst block, with a suggestion that trims it;
- give the rule: keep a call-site comment only if it says what breaks when the line changes; design reasoning goes once in a docstring or docs; drop history and numbers nothing keeps true;
- list **every** in-function block with a concrete one-line replacement.

Do not flag multi-line comments on declarations (BaseModel/dataclass config fields, class members, presets, module constants). Long docstrings are more acceptable, but should still be rare.

**Why:** The user, on PR #4052 (523 comment lines vs 1,769 code lines; one 33-line block inside a `with`): long mid-code comments are jarring, fill an agent's context ("if an agent can orient itself without the comment then we shouldn't add it"), and are re-edited in every later PR because they are wordy. Five were already wrong before merge. Cite those in the comment; they are the strongest evidence.
**How to apply:** run the block scan, classify by scope, cross-check each block against the other findings (a false comment is its own finding), and edit one comment rather than posting 20.

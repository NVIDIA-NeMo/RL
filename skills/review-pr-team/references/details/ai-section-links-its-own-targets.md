# ai-section-links-its-own-targets

> An ### AI-n that changes code off the anchor must link every target line INSIDE the AI section, even if the explanation above already linked them

When an action item asks for a change somewhere other than the comment's anchor ("pass it at both
`new_group` calls"), the AI section itself must list each target as a permalink — file:line plus the
enclosing function — and say what the line reads today. Links in the explanation paragraph above do
not count: the author reads the AI to find out what to DO, often alone, and a soft label like
"[policy]" / "[value]" two paragraphs up does not read as "here are the lines to edit".

**Why:** on PR #4481 the replica-group comment (C4) linked both `new_group` calls in its "how it
shows up" paragraph as "policy" / "value", then the AI said "Pass the same timeout at both
`new_group` calls" with no links. The user could not tell where the calls were. The audit passed
because its `perm` check only looked for any permalink per comment, not per AI section.
**How to apply:** in the pre-POST audit, for each `### AI-n` whose fix is not a suggestion block on
the anchor, require at least one RL permalink inside the AI section text, for every file it asks to
change. Also name the imports the fix needs (e.g. a helper from another module, `import datetime`).
Related: `permalink-must-support-the-sentence` (see [permalinks-and-anchors](../permalinks-and-anchors.md)), `verify-permalink-lines-programmatically` (see [permalinks-and-anchors](../permalinks-and-anchors.md)).
**Seen:** #4481 C4 (user)

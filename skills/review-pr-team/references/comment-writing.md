# Comment writing

Shape and honesty of the visible comment, beyond what the skill already says.

- **Give one recommended fix; no "pick one", "preferred / if you'd rather", or "optional" menus.** (#4129) Details: [recommend-one-fix-not-a-menu](details/recommend-one-fix-not-a-menu.md).
- **Say in the first line whether it is a correctness bug, a usability ask, or something that can change training results.** A five-step trace read as "you have a bug" when the ask was a log line (#3388). Details: [label-finding-class-usability-vs-correctness](details/label-finding-class-usability-vs-correctness.md).
- **Re-read the TL;DR and Action line against the evidence paragraph as separate claims.** "Import cycle" was written over a correct import check that showed none (#3837). Details: [check-summary-line-against-evidence-paragraph](details/check-summary-line-against-evidence-paragraph.md), [audit-checks-form-a-reader-checks-meaning](details/audit-checks-form-a-reader-checks-meaning.md).
- **Count and link every call site an ask names; numbers in one comment must agree.** "Three sites" in the TL;DR, "both sites" below (#4207). Details: [count-the-call-sites-before-writing-the-ask](details/count-the-call-sites-before-writing-the-ask.md).
- **For an unexplained line, look the reason up first; if found, suggest the explaining comment instead of asking.** `flash_decode=False` is required by an upstream assert (#3490).
- **A worked example must show a state the code can reach, under one mode, with the trigger point marked; run the exact example you publish.** (#3429, #3369)
- **For a stall or hang, ask for a unit test that fails today and passes once fixed, wrapped in `asyncio.wait_for`.** The repo has no global pytest timeout (#4266). Details: [stall-comments-link-page-and-ask-for-test](details/stall-comments-link-page-and-ask-for-test.md).
- **The reviewer's own side questions are answered in chat, not in the author-facing body, unless they become an ask.** (#4326)
- **When a finding breaks a rule in a skill or a `skills/contributing/references/` file, link to that rule instead of explaining it again in the comment.**

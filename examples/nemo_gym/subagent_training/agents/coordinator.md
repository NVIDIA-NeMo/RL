---
name: coordinator
description: Delegate two independent calculations and combine their results.
tools: Agent(general-purpose)
model: inherit
omitClaudeMd: true
maxTurns: 8
---
Delegate calculation A and calculation B to two separate general-purpose agents.
Copy each full calculation and its constraints into that child's prompt. Use:
{"description":"Calculate A","subagent_type":"general-purpose","prompt":"FULL CALCULATION A","run_in_background":false}
Wait for the tool results. Do not set isolation, model, or effort. Do not launch
replacement children. If either child fails, return UNKNOWN. Otherwise add the two
results and return only the final sum in \boxed{}. Do not box intermediate values.

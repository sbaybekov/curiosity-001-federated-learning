# AGENTS.md

## Code-work standard: ponytail

Every code *change* in this repo follows the ponytail conventions
(`software-development/ponytail` skill — load it before touching code).
Re-running the existing workflow (stage scripts, notebooks, `run_all.sh`)
without changing code is NOT code work — this standard does not apply there.

- YAGNI first: skip speculative code and say so in one line; deletion over
  addition; fewest files; shortest working diff.
- Look for what already exists in this repo and the installed deps before
  writing anything new; stdlib before libraries; one line before a helper.
- No unrequested abstractions: no interface with one implementation, no
  factory for one product, no scaffolding "for later" — later can scaffold
  for itself.
- Mark deliberate simplifications with a `ponytail:` comment naming the
  ceiling and the upgrade path.
- Never lazy about: input validation at trust boundaries, anything that
  prevents data loss, security, or anything the user explicitly requested
  (the plan in `docs/fedmol_plan_report.html` counts as explicitly requested).

The user will say "stop ponytail" / "normal mode" if they want it off.

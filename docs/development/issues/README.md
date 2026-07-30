# Hoosh Issues — How to File

Active issue reports live here. Resolved items move to `archived/`.

This directory follows the convention established in the **cyrius** repo
(`cyrius/docs/development/issues/`), so a consumer that files against both
ecosystems writes the same shape twice. `CLAUDE.md`'s work-loop step 0 already
says *"read roadmap, CHANGELOG, and open issues"* — this is where the third one
lives.

## What belongs here

- **Consumer-reported gaps** — a downstream project (agnosai, daimon, an AGNOS
  app) cannot do something across hoosh's HTTP surface, is working around it in
  production code right now, and the fix belongs in hoosh.
- **Bugs** — misleading errors, silent truncation, crashes, perf regressions.
- **Surface recommendations** — "we keep re-deriving this from what hoosh
  already knows; should the response just say so?"

## What doesn't belong here

- **Feature wishlists with no consumer stopgap.** Speculative work goes in
  `docs/development/roadmap.md` under the relevant section. The bar for an issue
  is: *someone is working around this in production code right now.*
- **hoosh's own asks on its dependencies.** Those already have a home —
  roadmap.md's `Upstream-gated (sandhi | ai-hwaccel | cyrius)` sections.
- **Upstream tool bugs.** File those upstream.

## How to file

Create `docs/development/issues/{YYYY-MM-DD}-{short-slug}.md`, kebab-case, with
the consumer's name in the slug when it is a specific project. Structure:

```markdown
# {title} — {short status}

**Status:** 🟡 **OPEN** — one line on why it is still open, and what you
verified and when.
**Placement:** the release it is pinned to, or "unpinned — backlog".
**Discovered:** YYYY-MM-DD during {context}
**Severity:** Low / Medium / High / Critical
**Affects:** hoosh {version range}

## Summary
## Reproduction
## Root cause (if known — speculation is fine, flag it as such)
## Proposed fix
## Consumer-side workaround (if any)
```

Put any runnable repro in `repros/`.

## Severity guide

- **Critical** — silent data corruption, security, broken bootstrap.
- **High** — hard failure on a shipping consumer, no workaround.
- **Medium** — hard failure with a workaround, or a silent correctness/cost gap.
- **Low** — misleading messages, doc mismatches, ergonomic papercuts.

## Lifecycle

When the fix lands, the file gets a `— RESOLVED` suffix in its heading, a status
paragraph pointing at the fix version and the CHANGELOG section that closed it,
and moves to `archived/`. The filename stays stable so links keep working.

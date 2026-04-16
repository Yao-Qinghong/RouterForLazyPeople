# Execute: ARCHITECTURE.md Doc Tightening

**Hand this file to your AI coding assistant.** It is self-contained.

---

## What this is

A 10-task plan to fix `docs/ARCHITECTURE.md`, `docs/API_SPEC.md`, and `README.md` so they accurately describe the current codebase and stop mixing implemented behavior with aspirational behavior. All tasks are documentation-only -- no code changes.

The full plan with code blocks and verification steps is at:
`docs/superpowers/plans/2026-04-16-architecture-doc-tightening.md`

## Instructions

1. Read the full plan file above first.
2. Execute each task (1 through 10) in order. Each task has numbered steps with checkbox syntax.
3. For every task: read the referenced code files to verify the documented behavior is accurate before writing. Do not copy the plan's markdown verbatim without checking it against the current code -- the plan was written against a snapshot and the code may have changed.
4. Commit after each task with the commit message shown in the plan.
5. After all 10 tasks, do a final read of all three docs to catch contradictions.

## Task summary

| # | File(s) | What to do |
|---|---|---|
| 1 | ARCHITECTURE.md :36-48, :111-141 | Add backend handoff/drain policy. Document `_evict_for_vram()` from `lifecycle.py:453`. Document that eviction is immediate (no drain). |
| 2 | ARCHITECTURE.md :145-160 | Document that capability filter and no-tier fallback are fail-open. List the risks. |
| 3 | ARCHITECTURE.md :52-89 | Add Model Identity section. Document `_slug()` from `discovery.py:94`, key sources, alias exposure, collision rules. |
| 4 | ARCHITECTURE.md (before appendices) | Add Open Questions table collecting all 12 known limitations currently buried in normative text. |
| 5 | ARCHITECTURE.md :275-294 | Add full `backends.yaml` schema table. Source: `BackendConfig` in `config.py:167-198`. |
| 6 | ARCHITECTURE.md :180-234 | Add llama.cpp day-2 gaps: slot alignment, stream-idle timeout, disconnect cancellation, repeated-failure backoff, health probe specificity. Mark each as not implemented. |
| 7 | ARCHITECTURE.md :237-272 | Add supported OpenAI parameter matrix and compatibility gaps (tool streaming, developer role, long prompts, usage field). |
| 8 | API_SPEC.md :1-97, README.md :24-36 | Add phase labels. Phase 1 = OpenAI chat + models + admin on llama.cpp. Phase 2+ = Anthropic, Gemini, embeddings, WS, catch-all, multi-engine. |
| 9 | ARCHITECTURE.md :374-385 | Rewrite engine appendix with concrete file paths and steps matching existing engine implementations. |
| 10 | All three files | Final grep for stale references: `request_timeout`, `Capabilities` (old name), `n_parallel` being set, `local=True`, wrong defaults. Fix any found. |

## Key code files to reference

| What | Where |
|---|---|
| BackendConfig + BackendCapabilities | `router/config.py:158-198` |
| `_infer_capabilities()` | `router/config.py:210-227` |
| Config validation | `router/config.py:558-621` |
| `_slug()` key generation | `router/discovery.py:94-98` |
| `_evict_for_vram()` | `router/lifecycle.py:453-500` |
| `ensure_running()` with VRAM check | `router/lifecycle.py:502-545` |
| `idle_watchdog()` | `router/lifecycle.py:621-663` |
| `mark_unhealthy()` (60s penalty) | `router/lifecycle.py:74-81` |
| `select_candidates()` fail-open filter | `router/routing.py:221-264` |
| No-tier fallback | `router/routing.py:237-241` |
| `build_llama_cmd()` (no --parallel) | `router/engines.py:216-232` |
| Semaphore + queue timeout | `router/proxy.py:296-312` (OpenAI), `:717-732` (Anthropic), `:989-995` (Gemini) |
| Health probe (any 200) | `router/lifecycle.py:116-132` |
| ProxyConfig defaults | `router/config.py:84-90` |

## Ground rules

- Every claim in the doc must match current code. If the plan says X and the code says Y, write Y.
- Mark unimplemented behaviors as "not yet implemented" or put them in the Open Questions table. Do not write aspirational specs as if they are current behavior.
- Do not change any code. This is docs-only.
- Commit after each task, not at the end.

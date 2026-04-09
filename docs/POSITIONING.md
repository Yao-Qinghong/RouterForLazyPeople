# Positioning And Release Status

This note is for maintainers and early testers. It explains how to describe the project, how it relates to nearby open-source tools, and what must be true before calling it ready for broad public use.

## Short Positioning

RouterForLazyPeople is a **DGX Spark-first local LLM router**.

It auto-discovers local models, starts one backend on demand, exposes OpenAI / Anthropic / Gemini-compatible endpoints, benchmarks real local speed, caches benchmark results for routing, and keeps beginner users away from bulk-start / OOM workflows.

## Current Status

This repo is in active pre-release development.

- Current focus: one operator-owned DGX Spark or similar NVIDIA CUDA Linux workstation
- Current safety stance: prefer one-model-at-a-time actions over automatic bulk model startup
- Automated tests exist for core routing, compatibility translation, API route wiring, benchmarking helpers, discovery helpers, config loading, metrics, docs consistency, and CLI regression helpers
- Full public-release QA is **not complete**
- Do not present this as production-hardened until the release checklist below is complete on clean hardware

## Related Projects

These projects solve adjacent problems. Do not claim RouterForLazyPeople is the first or only LLM router.

| Project | Overlap | Difference from this project |
|---|---|---|
| [llama-swap](https://github.com/mostlygeek/llama-swap) | OpenAI / Anthropic-compatible proxy that swaps local llama.cpp-style model servers | RouterForLazyPeople is aiming at DGX Spark onboarding, model discovery, status / sysinfo / bench CLI flows, benchmark-informed tier routing, and multi-engine lifecycle |
| [LiteLLM](https://github.com/BerriAI/litellm) | Mature OpenAI-compatible gateway, provider router, auth / spend / fallback features | LiteLLM is a provider gateway first; RouterForLazyPeople manages local backend processes and beginner local-model operations |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Quality / cost router between strong and weak LLMs | RouteLLM is router research / serving logic; RouterForLazyPeople is local workstation orchestration and compatibility proxying |
| [Ollama](https://github.com/ollama/ollama) | Beginner-friendly local model runner with an API | Ollama is its own runtime and model-management workflow; RouterForLazyPeople sits in front of existing local engines and local model files |
| [Jan](https://github.com/janhq/jan) | Local AI product with a user-facing app and local server | Jan is an application; RouterForLazyPeople is a headless router for connecting existing clients to local backends |
| [LocalAI](https://github.com/go-skynet/LocalAI) | OpenAI-compatible local inference server | LocalAI focuses on serving inference; RouterForLazyPeople focuses on routing, lazy lifecycle, diagnostics, benchmark cache, and compatibility proxying |

## Product Boundary

Say this:

> A beginner-oriented local LLM router for a DGX Spark-class machine. It exposes OpenAI / Anthropic / Gemini-compatible APIs, discovers local GGUF / HF-style models, starts selected backends lazily, stops idle backends, benchmarks one backend at a time, and uses cached local speed to improve routing.

Do not say this yet:

- "Production ready"
- "Fully QA tested"
- "Safe to bulk benchmark every discovered large model"
- "A replacement for LiteLLM"
- "A replacement for Ollama"
- "The first LLM router"

## Beginner UX Principles

- Show copy-paste commands, not only concepts.
- Show the exact backend key when the user needs to paste a model name.
- Show whether a command tests the router, a local backend server, or an external app.
- On DGX Spark, avoid flows that start many large models without an explicit flag.
- When a backend fails, print the backend log path and one likely next diagnostic command.
- When benchmark results are shown, explain how to use the key in a client model field or prompt-level route override.

## Public-Release QA Checklist

Before calling this ready for general open-source users, verify at least:

- Fresh-clone setup on target DGX Spark image
- Fresh-clone setup on one CUDA Linux workstation that is not the maintainer's daily machine
- Fresh-clone setup on macOS for llama.cpp / Metal, or document it as experimental
- Python 3.10+ guard fails clearly on too-old system Python
- `./router-start service install` installs, waits for router health, and points users at working logs
- `./router-start status` remains readable with long HuggingFace-derived backend keys
- `./router-start bench` does not start stopped discovered models by default
- `./router-start bench --backend KEY --start-stopped` starts, measures, saves, refreshes routing data, and stops the backend it started
- `./router-start bench --results` explains what model key to use next
- A failed vLLM / llama.cpp start prints a usable log path and a short likely diagnosis
- A known model too large for the device does not get auto-started by ordinary setup or benchmark prompts
- OpenAI chat, Anthropic messages, Gemini generateContent, websocket chat, `/v1/models`, `/status`, `/health`, `/benchmarks`, `/metrics`, and `/metrics/prometheus` smoke tests pass against a live router
- llama.cpp update success path is tested on CUDA
- llama.cpp update rollback path is tested with an intentionally failing build command
- README quick start is tested by someone who did not write the code
- API spec route-consistency test is green

## Release Note Template

Use explicit status language in public releases:

> This is an early DGX Spark-focused release. It is useful for local testing and dogfooding, but it has not yet been certified across the full engine / OS / GPU / client matrix. Benchmark one backend at a time and read the printed log path if a backend fails to start.

# Operations

This document is the operational spec for installation, startup, updates, supported environments, and day-2 maintenance.

## Supported Platforms And Engines

- Supported platforms: Linux and macOS
- Windows: not currently supported or tested
- Supported backend engines:
  - llama.cpp
  - vLLM
  - SGLang
  - TensorRT-LLM
  - HuggingFace TGI / transformers wrapper
  - Ollama

## Startup And Shutdown

- `python cli.py start` creates `.venv` on first run, installs requirements, and launches the router.
- `python cli.py stop` stops the router process and frees its port when possible.
- `python cli.py status` reports backend runtime state from the live router.
- `python cli.py logs` tails the router log file.

## Update Behavior

- `python cli.py update` refreshes llama.cpp and Python dependencies.
- `python cli.py update --restart` performs the update path and then restarts the router.
- llama.cpp rebuild mode is chosen from the local machine:
  - CUDA when a usable CUDA toolchain is detected
  - Metal on macOS
  - CPU-only otherwise

## Reload And Rescan Behavior

- `python cli.py rescan` calls the live router’s `/rescan` endpoint.
- `/rescan` updates the backend registry while leaving already-running backends alone.
- `/reload-config` reloads mutable router settings from the active config files without rebuilding middleware.

## Diagnostics And Monitoring

- `python cli.py sysinfo` works against the live router when available and falls back to local detection otherwise.
- `GET /metrics` provides aggregated request metrics.
- `GET /metrics/export` exports historical metrics as CSV.
- `GET /metrics/prometheus` exposes Prometheus-compatible metrics for scraping.

## Operational Limits

- Built-in rate limiting is not enforced. Use a reverse proxy or API gateway for throttling.
- The router does not manage TLS termination.
- Auth is optional and should not be treated as a full edge-security solution on untrusted networks.

## Recommended Operational Practice

- Keep the README focused on onboarding and usage examples.
- Treat `API_SPEC.md` as the public contract for clients and integrators.
- Treat `ARCHITECTURE.md` as the internal technical source of truth.
- Treat this document as the runbook for install, update, reload, and monitoring behavior.

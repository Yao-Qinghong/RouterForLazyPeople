# Operations

This document is the operational spec for installation, startup, updates, supported environments, and day-2 maintenance.

## First Run

Default DGX Spark / NVIDIA workstation path:

```bash
git clone https://github.com/Yao-Qinghong/RouterForLazyPeople
cd RouterForLazyPeople
python3 --version      # must be 3.10+
./router-start start
./router-start status
```

What happens on first run:

- The CLI verifies that the selected Python is 3.10 or newer
- A local `.venv` is created if missing
- Python dependencies are installed
- The router starts on the configured port
- Common model directories are scanned automatically

When manual config is needed:

- Edit `config/backends.yaml` if you want fixed `fast` / `mid` / `deep` backends
- Edit `config/settings.yaml` if your models live outside the default `scan_dirs`
- After changing scan directories while the router is running, call `./router-start rescan`

After backends appear in `./router-start status`, benchmark one model at a time:

```bash
./router-start bench --backend <backend-key> --start-stopped
```

This starts the named backend, measures prompt-processing (PP) and token-generation (TG) speed, saves results under the configured data directory, asks the live router to rescan so speed-informed routing can use the new cache immediately, then stops the backend if bench started it.

After one or more successful runs, use `./router-start bench --results` to see cached results, copy the fastest measured backend key into a client model field, or copy the printed `[route:key]` prompt override. Saved TG speed improves automatic selection among backends that are already assigned to the same configured tier; it does not automatically change a backend from `deep` to `fast`.

DGX Spark safety: `python cli.py bench` measures currently running backends by default. It does not bulk-start discovered models. Use `--backend <key> --start-stopped` for one stopped model, or `--all --start-stopped` only when you have verified that every discovered backend can fit.

## Supported Platforms And Engines

- Supported platforms: Linux and macOS
- Windows: not currently supported or tested
- Default engine: **llama.cpp** (the only engine enabled out of the box)
- Additional engines (require `engines_enabled` in `settings.yaml`):
  - vLLM
  - SGLang
  - TensorRT-LLM
  - HuggingFace TGI / transformers wrapper
  - Ollama

### Enabling Additional Engines

Add the engine name to `engines_enabled` in `config/settings.yaml`:

```yaml
engines_enabled:
  - "llama.cpp"
  - "vllm"
```

Then call `/rescan` or restart the router. Engines not listed are ignored — their binaries are not probed, their models are not discovered, and they cannot be started.

## Startup And Shutdown

- `python cli.py start` creates `.venv` on first run, installs requirements, and launches the router.
- `./router-start start` is the preferred beginner entrypoint; it picks the project `.venv` first and otherwise searches common Python 3.10+ executable names.
- `python cli.py stop` stops the router process and frees its port when possible.
- `python cli.py status` reports backend runtime state from the live router.
- `python cli.py logs` tails the router log file.
- Default client endpoints after startup:
  - OpenAI-compatible: `http://localhost:9001/v1`
  - Anthropic-compatible: `http://localhost:9001/anthropic`
  - Model discovery: `http://localhost:9001/v1/models`

## Update Behavior

- `python cli.py update` refreshes llama.cpp and Python dependencies.
- `python cli.py update --restart` performs the update path and then restarts the router.
- llama.cpp source is pulled with fast-forward-only Git semantics.
- Before rebuilding, an existing configured `llama-server` binary is copied to a temporary backup.
- If the pull/configure/build path fails, the CLI checks llama.cpp back out to the previous commit and restores the backed-up binary.
- If a rollback message appears, keep running the old router, fix the CUDA/CMake/toolchain problem, then retry `python cli.py update`.
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
- `python cli.py bench` is an active speed test for currently running managed backends. It sends fixed prompts, caches PP/TG speeds, and refreshes router routing data.
- `python cli.py bench --backend <key> --start-stopped` intentionally starts one stopped backend for measurement and stops it afterward unless `--keep-running` is supplied.
- `python cli.py bench --results` shows cached benchmark results, the best measured backend key, and a prompt-level route override example.
- `python cli.py benchmark` is a passive metrics view. It reports request metrics from traffic the router has already served.
- `GET /benchmarks` returns the cached active speed-test results.
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
- Keep docs structure details in `docs/README.md`, not in the middle of the README quick-start flow.

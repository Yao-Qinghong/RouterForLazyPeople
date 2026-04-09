# Docs Index

This repo separates onboarding, public behavior, internal design, and operations into distinct documents:

- [`../README.md`](../README.md): quick-start, product overview, and usage-oriented entry point

- [`API_SPEC.md`](API_SPEC.md): public/functional behavior, supported endpoints, routing rules, auth behavior, and compatibility surfaces
- [`ARCHITECTURE.md`](ARCHITECTURE.md): internal technical design, component boundaries, data flow, and configuration precedence
- [`OPERATIONS.md`](OPERATIONS.md): install/update/reload behavior, supported platforms and engines, and operational non-goals

Guideline:

- Keep `README.md` short and biased toward getting the router running
- Put public route and behavior guarantees in `API_SPEC.md`
- Put internal implementation and module boundaries in `ARCHITECTURE.md`
- Put installation, update, reload, and monitoring runbooks in `OPERATIONS.md`

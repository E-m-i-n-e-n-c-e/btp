# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo currently contains a single project: `trace-fl/`. All commands below are run from `trace-fl/`.

TRACE-FL is a federated-learning research platform built on Flower (`flwr`) + PyTorch. Phase 1 (the current phase) establishes the FL foundation: simulated clients training on MNIST partitions, FedAvg aggregation, and telemetry — as a base for later phases (attack injection, CKA-based malicious-update detection, trust-aware aggregation). The root `README.md` is the Phase 1 spec (team split, planned attacks/aggregators/metrics, and an explicit "not in Phase 1" list — check it before adding scope like temporal trust, PKI, or Sybil detection). Its "Repository Structure" section is aspirational; the actual package layout is `app/`, `core/`, `data/`, `federation/`, `telemetry/`, `training/` as described below. `trace-fl/TRACE_FL_WALKTHROUGH.md` walks through the round lifecycle.

## Commands

All commands run from the `trace-fl/` directory using `uv` (required — this project standardizes on `uv` for reproducible envs, pinned to Python 3.11.14 via `.python-version`).

```bash
# Install dependencies (pytest is in the `dev` extra)
uv sync --extra dev

# Run the FL simulation — use the script, not bare `flwr run .` (see below)
uv run sh scripts/run_simulation.sh

# Run tests
uv run pytest
uv run pytest tests/test_model.py            # single test file
uv run pytest tests/test_model.py::test_name # single test case

# Sanity-check the local environment (Python version, package versions)
uv run python scripts/check_environment.py

# Docker (see trace-fl/DOCKER.md): same commands inside a pinned Linux container
python run.py                                    # interactive launcher (stdlib-only, runs on host); per-run config overrides are mounted from experiments/<timestamp>/config.yaml, never written to config/config.yaml unless the user confirms
python run.py sim | deploy | test [pytest args] | shell | rebuild   # deploy = SuperLink + N SuperNode containers
docker compose run --rm --build trace-fl pytest  # what `run.py test` wraps
```

Simulation parameters (client count, rounds, local epochs, batch size, LR, seed) live in `config/config.yaml` and are read directly by both `app/server_app.py` and `app/client_app.py` at runtime — not via Flower's `run_config`/`pyproject.toml` toml tables.

Flower 1.37 quirks that `scripts/run_simulation.sh` handles:
- `flwr run .` auto-starts a background local SuperLink that defaults to **2** simulated SuperNodes. The script runs `flwr federation simulation-config --num-supernodes <num_clients>` first. A mismatch either leaves partitions unused or crashes `get_partition` on an out-of-range `partition-id`.
- By default that SuperLink re-resolves app dependencies into a fresh env per run, ignoring `uv.lock`. `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` turns this off.
- The app is loaded from a FAB (Flower App Bundle) built with `.gitignore` rules applied. Don't add gitignore patterns that match source directories; `data/` is a Python package, and only `/data/MNIST/` is ignored.

## Architecture

### Flower app entry points

`pyproject.toml` registers the Flower app components: `app.server_app:app` (ServerApp) and `app.client_app:app` (ClientApp). These are the modules Flower's simulation engine (Ray-backed) actually invokes.

### Round lifecycle (server side: `app/server_app.py`)

`TRACEStrategy` (implements Flower's `Strategy` interface) drives each round:
- `configure_fit` — samples clients via `federation/client_selector.py::RandomClientSelector`, tracks downlink bytes, sends current global `Parameters`.
- `fit` happens client-side (see below); results come back as `(ClientProxy, FitRes)` pairs.
- `aggregate_fit` — logs each client's reported loss/accuracy explicitly via `logger.info` (this bypasses Ray's log deduplication, which otherwise hides per-actor logs), tracks uplink bytes, validates each update with `federation/update_validator.py::UpdateValidator` (example count > 0, float32, finite, correct shapes, plus optional `max_num_examples` / `max_update_norm` caps), logs per-client update signals, then aggregates valid updates with the configured aggregator (default FedAvg, num-examples-weighted).
- `configure_evaluate` — intentionally returns `[]`; **federated (client-side) evaluation is disabled**. Do not "fix" this without checking intent — the design evaluates the global model centrally instead (see below).
- `evaluate` — server evaluates the aggregated global model centrally against the **full** MNIST test set (`FederatedDatasetProvider.get_global_test_loader()`), not against per-client local splits. This is deliberate: it gives an unbiased accuracy signal independent of any client's local data. Emits one structured `round_complete` JSON log line via `telemetry/logging.py::log_round`.

Per-round bookkeeping (selected/successful/failed clients, timings, global loss/accuracy) lives in `federation/round_state.py::RoundState`, reset at the start of each round.

### Client side (`app/client_app.py`)

`TRACEClient` (Flower `NumPyClient`) loads its own data partition via `data/dataset.py::FederatedDatasetProvider`, trains locally (`training/trainer.py::train`), and returns updated parameters + example count + a metrics dict containing `client_id` (used server-side to identify the client in logs, since Ray's `ClientProxy.cid` isn't human-readable).

**Partition ID assignment.** `context.node_id` is a random 64-bit int, so it can't be used to pick a data partition. `client_fn` uses `context.node_config["partition-id"]`, which Flower's simulation provides (0..num-supernodes-1). In deployment mode `run.py` passes it per SuperNode via `--node-config "partition-id=i num-partitions=N"`.

**Config loading.** Always use `core/config.py::load_config()`, which resolves `config/config.yaml` relative to the code rather than the CWD. Flower loads the app from an installed FAB, and the CWD differs between simulation and deployment containers.

### Data partitioning (`data/dataset.py`, `data/partition.py`)

`FederatedDatasetProvider` downloads MNIST into `./data`, then uses `torch.Generator().manual_seed(seed)` with `random_split` to deterministically split both train and test sets into `num_clients` equal-ish `LocalPartition`s (IID split — the non-IID/Dirichlet partitioning described in the Phase 1 plan is not yet implemented). Because the split is seeded, it's reproducible across the separate client processes in the Ray simulation without any inter-process coordination beyond the shared seed.

### Telemetry (`telemetry/`)

- `telemetry/communication.py::CommunicationTracker` — estimates serialized payload size (`nbytes` sum over NumPy arrays) for every uplink/downlink transfer, tracked per-client and per-round. Instantiated independently on both the server and each client (separate processes), so totals are reconciled/logged from the server side in `aggregate_fit`/`evaluate`.
- `telemetry/logging.py` — emits structured JSON log lines (`client_complete`, `client_evaluate`, `round_complete` events) through a shared `TRACE-FL` logger. When adding new telemetry, follow this pattern (a `log_*` function that builds a dict and calls `logger.info(json.dumps(...))`) rather than ad hoc log statements, since downstream analysis expects structured JSON events.

### Core model/parameter utilities (`core/`)

- `core/model.py` — `MNISTNet` (small CNN: 2 conv+pool blocks, fc1=128 "penultimate" layer exposed via `get_penultimate_layer` for future CKA-based representation analysis, fc2=10 classes) plus `get_parameters`/`set_parameters` for converting between PyTorch state dicts and Flower's NumPy-array parameter representation.
- `core/parameters.py` — thin wrappers around `flwr.common.ndarrays_to_parameters`/`parameters_to_ndarrays`.
- `core/types.py` — shared `NDArrays` type alias.

### Extension points implied by the architecture

- `federation/strategy.py::AggregationStrategy` has `FedAvgStrategy`, `MedianStrategy`, `TrimmedMeanStrategy`, selected by `federation.aggregation` via `build_aggregation_strategy`; Krum (Member 4) would be added to the `AGGREGATORS` registry.
- `federation/client_selector.py::ClientSelector` is similarly an abstract interface with only `RandomClientSelector` implemented.
- `attacks/` is the attack engine (label flipping, Gaussian noise, sign flip, scaling), attached to `TRACEClient.fit` via two hooks: `apply_pre_training` (wraps the DataLoader) and `apply_post_training` (rewrites the update). Configured by the `attack:` block; see `trace-fl/docs/ATTACKS.md`.
- `federation/update_stats.py::compute_update_stats` (update norm, cosine to the leave-one-out median of peers) is logged per client as a `client_update` event. It is measurement only; CKA is not implemented yet (`core/model.py::get_penultimate_layer` exists for it). The client-reported `is_attacked` flag is ground truth for analysis and must never be read by a defense.
- `scripts/plot_attack_benchmark.py` runs attack sweeps in-process through the same components (no Flower/Ray) and writes `artifacts/benchmarks/*.png`; `--demo <attack>` is a ~1 min live check.

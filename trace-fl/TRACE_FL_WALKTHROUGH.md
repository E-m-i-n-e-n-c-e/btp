# TRACE-FL: Architecture & Codebase Walkthrough

This document provides a technical walkthrough of the TRACE-FL federated learning framework. It covers how to run the system and explains the underlying architecture, data partitioning, and round lifecycle without any extraneous Q&A.

## 1. Quickstart & Installation

This project strictly uses `uv` for fast, reproducible dependency management, ensuring identical environments across all machines.

### Sync Dependencies
Ensure you have `uv` installed, then run the following in the project root to install all dependencies from `pyproject.toml` into an isolated virtual environment:
```bash
uv sync
```

### Run the Simulation
To execute a local federated learning simulation (`num_clients` from `config/config.yaml`), use the following command:
```bash
uv run sh scripts/run_simulation.sh   # sets the SuperNode count to num_clients first
```
*Note: The `--stream` flag ensures real-time telemetry logs are output directly to your terminal.*

---

## 2. Telemetry and Logging

TRACE-FL utilizes a custom `CommunicationTracker` that outputs highly structured JSON events. 
By pulling the logs up to the central server during the `aggregate_fit` phase, the framework successfully bypasses the log deduplication native to distributed systems (like Ray). 

You will see raw logs like this for each client:
```text
INFO:TRACE-FL:Client client_0 local training complete -> Loss: 0.2111 | Accuracy: 0.9373
```

---

## 3. The Lifecycle of a Round

The server orchestrates the learning process using a custom strategy (`TRACEStrategy` in `app/server_app.py`). Here is the exact lifecycle of a single round:

### A. `configure_fit` (Server-Side)
The server begins the round by selecting a subset of available clients. It sends the current global model weights to these selected clients.

### B. `fit` (Client-Side)
Each selected client wakes up, receives the global weights, and applies them to its local model. The client then trains the model on its isolated data partition.
Once finished, the client returns:
1. The new, updated model weights.
2. The number of examples it trained on.
3. A dictionary containing its local `loss`, `accuracy`, and human-readable `client_id`.

### C. `aggregate_fit` (Server-Side)
The server waits for all selected clients to finish their local training. It then performs **Federated Averaging (FedAvg)**, mathematically averaging the incoming model weights (weighted by how much data each client had) to produce the new global model.

### D. `evaluate` (Server-Side)
Once the new global model is built, the server tests it.
Crucially, **the server evaluates the model centrally against the full 10,000-image global test dataset**.
This guarantees a completely unbiased metric of how well the aggregated model performs globally, completely independent of the clients' local training sets. The results are outputted in the `round_complete` telemetry event.

---

## 4. Ensuring Data Partitioning

A core requirement of Federated Learning is that clients only train on their own, distinct data. TRACE-FL mathematically guarantees this through deterministic splitting.

### `FederatedDatasetProvider`
In `data/dataset.py`, the `FederatedDatasetProvider` uses a fixed random seed (`torch.Generator().manual_seed(seed)`) in conjunction with PyTorch's `random_split`. Because the seed is fixed, the split is perfectly deterministic across all client processes.

### Resolving the `partition_id`
Flower's simulation gives every virtual SuperNode a `partition-id` (0..N-1) in `context.node_config`; in deployment mode `run.py` passes it with `--node-config "partition-id=i num-partitions=N"`. `client_fn` reads it and refuses to start without it (`context.node_id` is a random 64-bit integer, so `node_id % N` could hand two clients the same partition). Client `i` therefore loads `train_partitions[i]`, and partitions are mutually exclusive.

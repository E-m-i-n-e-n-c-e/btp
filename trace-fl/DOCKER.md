# Running TRACE-FL in Docker

TRACE-FL runs in Docker so that every machine (macOS on Intel or Apple Silicon, Linux, Windows) uses the same Python, the same library versions and the same dataset. There are two modes:

- **Simulation** (default): one container runs the Flower server and all clients as a simulation.
- **Deployment** (optional): the server and each client run in separate containers and communicate over gRPC.

## Quickstart

You need Docker Desktop (macOS or Windows) or Docker Engine with the Compose plugin (Linux), plus any Python 3 to run the launcher.

```bash
cd trace-fl
python run.py        # python3 on some Macs/Linux, py on Windows
```

`run.py` checks that Docker is installed and running, then shows a menu:

```
TRACE-FL (Docker)
  1) Run simulation
  2) Run tests
  3) Open a shell in the container
  4) Rebuild image from scratch (only if something seems broken)
  q) Quit
```

- **Run simulation** shows the default settings from `config/config.yaml` (clients, rounds, dataset name, partition type, Dirichlet alpha, epochs, batch size, learning rate, seed) and lets you change any of them **for that run only**. It then asks whether to use deployment mode (default: no).
  - `config/config.yaml` is left unchanged. After the run you're asked whether to save the changes as the new defaults (default: no).
  - Every run records the exact config it used in `experiments/<timestamp>/config.yaml` (git-ignored).
- Every option rebuilds the image first if the code or dependencies changed. The first build downloads PyTorch, MNIST, and Fashion-MNIST and takes a few minutes; later runs start in seconds.

`run.py` uses only Python's standard library. It also accepts direct commands with optional split/training flags:

| Task | `run.py` | Plain Docker (no Python on host) |
|---|---|---|
| Run simulation (defaults) | `python run.py sim` | `docker compose run --rm --build trace-fl` |
| Run with custom split | `python run.py sim --dataset mnist --partition iid` | (edit `config/config.yaml` then run compose) |
| Run with Dirichlet Non-IID | `python run.py sim --dataset fashion_mnist --partition dirichlet --alpha 0.1` | (edit `config/config.yaml` then run compose) |
| Run in deployment mode | `python run.py deploy` | (requires `run.py`, which generates the compose file) |
| Run all tests | `python run.py test` | `docker compose run --rm --build trace-fl pytest` |
| Run one test file | `python run.py test tests/test_model.py` | `docker compose run --rm --build trace-fl pytest tests/test_model.py` |
| Shell inside the container | `python run.py shell` | `docker compose run --rm --build trace-fl bash` |
| Rebuild from scratch | `python run.py rebuild` | `docker compose build --no-cache` |

To change the defaults permanently, edit `config/config.yaml`. `config/` is mounted into the container, so edits take effect without a rebuild. Supported dataset splits (see `DatasetSplit_Documentation.pdf`):
- `dataset.name`: `mnist` or `fashion_mnist`
- `dataset.partition_type`: `iid` or `dirichlet`
- `dataset.alpha`: `1.0` (mild skew), `0.5` (moderate skew), `0.1` (strong skew)

## Simulation mode

```
┌──────────────────────── trace-fl container ────────────────────────┐
│  scripts/run_simulation.sh                                         │
│    1. flwr federation simulation-config --num-supernodes <N>       │
│    2. flwr run . --stream                                          │
│                                                                    │
│  flwr run ──starts──► local SuperLink (background, simulation mode)│
│                          │                                         │
│                          ▼                                         │
│                   Simulation runtime (Ray)                         │
│                   ServerApp ── TRACEStrategy (FedAvg)              │
│                      │                                             │
│          ┌───────────┼───────────┐                                 │
│      ClientApp 0  ClientApp 1 … ClientApp N-1                      │
│      partition 0  partition 1   partition N-1                      │
│                                                                    │
│  /opt/venv        locked Python 3.11.14 environment (from uv.lock) │
│  /app             project source                                   │
│  /app/data/MNIST  dataset, downloaded when the image is built      │
└────────────────────────────────────────────────────────────────────┘
```

Clients are virtual and scheduled by Ray, so memory use depends on the CPU count rather than the number of clients. By default Flower gives each simulated client 2 CPUs, so about (Docker CPUs ÷ 2) clients train at the same time. For large experiments (e.g. 20 clients), give Docker Desktop more CPUs and memory under Settings → Resources.

## Deployment mode

```
          docker network (gRPC)
┌─────────────┐              ┌──────────────┐
│  superlink  │◄────────────►│ supernode-0  │  partition-id=0
│ (ServerApp) │◄────────────►│ supernode-1  │  partition-id=1
│             │◄────────────►│ supernode-N-1│  ...
└─────────────┘              └──────────────┘
       ▲
       │ flwr run . deploy
┌─────────────┐
│     cli     │
└─────────────┘
```

- **superlink:** the server; it runs the ServerApp.
- **supernode-i:** one per client; it runs the ClientApp on partition `i`, set with `--node-config "partition-id=<i> num-partitions=<N>"`.
- **cli:** submits the run and checks which clients are online.

Use deployment mode for anything that needs real network behaviour: stopping a client mid-run, running an attacker as its own container, measuring real communication, or node authentication. Keep large experiments on simulation.

Start it from the menu or with `python run.py deploy`. `run.py`:

1. Estimates memory (~1.5 GB + ~0.6 GB per client) against Docker's limit and warns if it won't fit.
2. Generates `experiments/<timestamp>/docker-compose.deploy.json`.
3. Starts the containers and waits until all N clients are online.
4. Runs `flwr run . deploy --stream`, which streams the server logs.
5. Saves every container's logs to `experiments/<timestamp>/containers.log` and removes the containers.

Notes:
- **Memory:** each client container loads its own PyTorch and MNIST, so memory grows with the client count. With Docker's default 8 GB, keep to about 8 clients or fewer.
- **Encryption:** connections use `--insecure` (no TLS). That's suitable for a single machine, not for real networks.
- **Client logs:** per-client `client_complete` events are in `containers.log`. The live stream shows the server side, which logs every client's loss and accuracy.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | `python:3.11.14-slim-bookworm` + `uv` 0.12.19; installs `uv.lock` with `uv sync --frozen`; downloads MNIST at build time |
| `docker-compose.yml` | Simulation service: sets `shm_size` for Ray; mounts `config/` (read-only) and `experiments/` |
| `run.py` | Launcher: checks Docker, per-run config overrides, simulation and deployment modes |
| `scripts/run_simulation.sh` | Sets the simulated client count to `num_clients` and starts the run; also works outside Docker |
| `core/config.py` | Loads `config/config.yaml` relative to the code, so it works in every container |
| `.dockerignore` | Keeps the host's `.venv`, caches and downloaded data out of the image |
| `.gitattributes` | Forces LF line endings so shell scripts work when cloned on Windows |

## Design notes

Each of these settings prevents a failure, or a silent difference in results, between machines.

- **Pinned environment.** The image uses exactly Python 3.11.14 (as `pyproject.toml` requires) and installs dependencies with `uv sync --frozen`, so every run uses the versions in `uv.lock`. Host Python versions don't matter.
- **No per-run dependency installation.** By default, Flower's SuperLink creates a fresh environment for each run and resolves dependencies from `pyproject.toml`, ignoring `uv.lock`. That can pull different versions (e.g. numpy 2.x despite `numpy<2`) and needs internet access. `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1`, set in the image and in `scripts/run_simulation.sh`, keeps runs on the locked environment.
- **Client count matches the config.** Flower (≥1.32) starts 2 simulated clients unless told otherwise:
  - with more partitions than clients, the extra partitions never train, and nothing reports it;
  - with fewer partitions than clients, a client asks for a partition that doesn't exist and crashes.

  `scripts/run_simulation.sh` sets the count from `num_clients` before each run.
- **Dataset baked into the image.** MNIST is downloaded at build time, so runs need no network and simulated clients never race to download into the same folder.
- **Shared memory for Ray.** Ray's object store lives in `/dev/shm` and wants more than 30% of RAM there, while Docker's default is 64 MB. The compose file sets `shm_size: 3gb`. Raise it if Docker has more than about 10 GB of RAM.
- **Paths independent of the working directory.** Flower loads the app from an installed bundle (FAB), and the working directory differs between modes. The config is loaded via `core/config.py` relative to the code, and clients get their partition from Flower's `partition-id`.
- **Bundle contents follow `.gitignore`.** `flwr run` leaves `.gitignore`d files out of the FAB. `data/` is a Python package, so only `/data/MNIST/` is ignored. Don't add ignore patterns that match source directories.
- **Line endings.** `.gitattributes` forces LF, so shell scripts still run inside Linux containers when the repo is cloned on Windows.
- **CPU architecture.** `uv.lock` includes CPU PyTorch wheels for `linux/arm64` (Apple Silicon) and `linux/amd64` (Intel/AMD), so the image builds natively on both.
- **CPU only.** Docker on macOS can't use the Apple GPU, and the lockfile pins CPU-only PyTorch. The code uses `cuda:0` only if it's available, so runs are CPU-based everywhere. That's sufficient for MNIST-scale experiments.

## Running without Docker

With `uv` installed:

```bash
cd trace-fl
uv sync --extra dev
uv run sh scripts/run_simulation.sh   # instead of bare `flwr run .`; see Design notes
uv run pytest
```

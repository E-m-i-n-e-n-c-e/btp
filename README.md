# TRACE-FL — Phase 1: FL Foundation + Attack Benchmark

## 1. Goal

Phase 1 establishes the **basic federated-learning experiment platform** that every later TRACE-FL component will use.

By the end of Phase 1, the team should be able to:

```text
Create clients
    ↓
Give every client different local data
    ↓
Train locally
    ↓
Send model updates
    ↓
Aggregate with FedAvg
    ↓
Repeat for multiple rounds
    ↓
Inject attacks into selected clients
    ↓
Measure whether malicious updates look different

# 2. Technology Choice

We will use:

* Python
* PyTorch
* Flower
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

For Phase 1, we will use **Flower Simulation**.

We will NOT initially deploy separate physical servers or cloud machines.

A single machine will simulate the federated system using multiple virtual clients.

```text
                 ONE MACHINE

              Flower ServerApp
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     Client 1    Client 2    Client 3
        ↓           ↓           ↓
      Data 1      Data 2      Data 3

              ... up to 20 clients
```

---

# 3. What "Server" Means

The **Flower ServerApp** is simply our server-side federated-learning logic.

It is responsible for:

* starting federated rounds
* selecting clients
* sending the current global model
* receiving client updates
* aggregating updates using FedAvg
* producing the next global model

We are NOT writing a custom socket server.

Flower provides the federation runtime.

Our code mainly defines what the server should do.

---

# 4. What "Client" Means

A **Flower ClientApp** represents one participating FL client.

Each client owns its own local dataset.

For example:

```text
Client 1 → local dataset D1
Client 2 → local dataset D2
Client 3 → local dataset D3
...
Client 20 → local dataset D20
```

Each client:

1. receives the current global model
2. trains locally
3. produces an updated model
4. sends the update back

The local dataset never needs to be combined into one centralized training dataset.

---

# 5. Phase 1 Architecture

```text
                    PHASE 1
              FLOWER SIMULATION

                 ServerApp
                    │
             Global Model
                    │
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
 ClientApp 1   ClientApp 2   ClientApp N
      │             │             │
     D1            D2            DN
      │             │             │
 Local Training  Local Training  Local Training
      │             │             │
      └─────────────┼─────────────┘
                    ↓
             Client Updates
                    │
                    ▼
               ServerApp
                    │
                  FedAvg
                    │
                    ▼
              Global Model
                    │
                    ▼
                Next Round
```

---

# 6. Team Split

## Member 1 — Flower + FL Infrastructure

### Main responsibility

Build the federated-learning skeleton.

### Implement

* Flower project
* `ServerApp`
* `ClientApp`
* FedAvg
* round lifecycle
* model parameter exchange
* client registration/identification
* basic logging
* communication byte counter

### Important clarification

Member 1 is NOT building a FastAPI backend yet.

The objective is:

```text
Flower ServerApp
        +
Flower ClientApps
        +
working FL rounds
```

### Deliverable

A working command that can run:

```text
20 clients
→ local training
→ FedAvg
→ 20 rounds
→ final global model
```

---

# 7. Member 2 — Dataset + Non-IID Pipeline

### Main responsibility

Create realistic federated client datasets.

### Initial datasets

Start with:

```text
MNIST
Fashion-MNIST
```

### Partitioning

Implement:

### IID

Every client gets approximately the same class distribution.

```text
Client 1 → balanced
Client 2 → balanced
Client 3 → balanced
```

### Non-IID

Use Dirichlet partitioning:

```text
α = 1.0
α = 0.5
α = 0.1
```

Smaller α means stronger heterogeneity.

Example:

```text
Client 1 → mostly class 1
Client 2 → mostly class 7
Client 3 → classes 2, 5
Client 4 → classes 3, 8
```

### Deliverables

* dataset loaders
* partition generator
* reproducible random seeds
* client dataloaders
* class-distribution plots

---

# 8. Member 3 — Attack Engine

### Main responsibility

Create malicious clients without modifying the core FL system.

The attack module should behave like a plug-in.

```text
Normal Client
      ↓
Local Training
      ↓
Update

Attacker Client
      ↓
Local Training
      ↓
Attack
      ↓
Malicious Update
```

### Initial attacks

Implement:

```text
1. Label Flipping
2. Gaussian Noise
3. Sign Flip
4. Scaling / Boosting
```

### Attack configuration

Every attack should be configurable.

Example:

```yaml
attack:
  enabled: true
  type: sign_flip
  intensity: 1.0
  start_round: 5
  end_round: 15
```

Also support:

```text
continuous attacker
intermittent attacker
```

Example:

```text
Round 1 → honest
Round 2 → honest
Round 3 → attack
Round 4 → honest
Round 5 → attack
```

### Deliverable

A reusable:

```text
AttackEngine
```

that can be attached to any client.

## 8.1 Status: implemented, with measured results

Full write-up (architecture, threat model, the math of each attack, every
command, all results, vulnerabilities): **[`trace-fl/docs/ATTACKS.md`](trace-fl/docs/ATTACKS.md)**.

### Where the attacks plug in

```text
client fit():  w_global ─► [HOOK 1: data poisoning]  ─► local SGD ─► [HOOK 2: model poisoning] ─► send
                           label_flipping wraps the        Δ = w_local − w_global is rewritten:
                           DataLoader (y -> 9-y,           gaussian  Δ + N(0, σ²)
                           or 7 -> 1 targeted)             sign_flip −γΔ
                                                           scaling    βΔ
server:  validate (float32, finite, shapes, [opt] norm bound, [opt] num_examples cap)
         ─► log norm + cosine-to-peers per client ─► fedavg | median | trimmed_mean ─► evaluate on 10k test set
```

Honest and malicious clients run the same code; `AttackEngine.from_config`
is a no-op unless `attack.enabled: true`, and only targeted clients attack.

### Accuracy drop per attack

MNIST, 10 clients, **2 malicious (20%)**, FedAvg, 10 rounds, seed 42.

| Attack | Intensity | What the attacker sends | Final accuracy | Drop |
|---|---|---|---|---|
| None | — | honest update | **96.78%** | — |
| Label flipping | 100% of labels | gradient of the wrong labels | 96.52% | −0.3 pp |
| Gaussian noise | σ = 300 × update RMS | Δ + noise | **12.86%** | −83.9 pp |
| Sign flip | γ = 4 | −4Δ | **10.30%** | −86.5 pp |
| Scaling | β = 10 | 10Δ | **28.08%** | −68.7 pp |

![Accuracy per round under each attack](trace-fl/artifacts/benchmarks/attack_accuracy_vs_round.png)

![Final accuracy vs share of malicious clients](trace-fl/artifacts/benchmarks/attack_fraction_sweep.png)

![Final accuracy vs attack intensity](trace-fl/artifacts/benchmarks/attack_intensity_sweep.png)

Key findings:

1. **Sign flip has a closed-form breaking point.** FedAvg's step is
   `(1 − f(1+γ))·Δ̄`, so training stalls at γ* = (1 − f)/f = 4 for f = 0.2.
   Measured: γ = 2 gives 93.5%, γ = 4 gives 10.3%, and γ = 8 gives 1.2% (below chance).
2. **Gaussian noise is a cliff.** It is harmless up to 100× the update size
   (random noise in 421k dimensions averages out) and destructive at 300×.
3. **Label flipping is weak against FedAvg** until attackers reach 50% of
   clients (47.5%; 46% of test digits are then predicted as 9 − y). Its
   *targeted* form hides inside global accuracy: 7 → 1 flipping costs only
   0.55 pp globally but drops class-7 recall from 95.7% to 88.1%.
4. **Scaling is invisible to cosine** (same direction as honest clients) and
   visible only by norm. Sign flip is the opposite. A detector needs both signals.

### Do malicious updates look different?

![Per-client update norm vs cosine to peers](trace-fl/artifacts/benchmarks/attack_update_signatures.png)

Under IID every attack separates cleanly. Under Dirichlet α = 0.5 (bottom
row) the honest clients spread out, and label flipping's margin narrows. This is the
Phase 1 question for CKA (Section 13).

### Baseline defenses

![Final accuracy by aggregation rule](trace-fl/artifacts/benchmarks/attack_defense_matrix.png)

Median and trimmed mean recover every attack (≥ 96.4%). A static norm bound
recovers noise and scaling but **only partly recovers sign flip (87.3%)**:
honest update norms shrink from 1.59 to 0.23 over training, so from round 3
the attacker fits under the fixed bound.

### Commands

```bash
cd trace-fl

# ~1 minute: one attack vs no attack, prints accuracy per round + update signature
uv run python scripts/plot_attack_benchmark.py --demo sign_flip      # or label_flipping | gaussian_noise | scaling

# Real Flower runtime in Docker (config/config.yaml is not modified)
python run.py sim --dataset mnist --partition iid --clients 10 --rounds 10 \
  --attack-type sign_flip --attack-intensity 4 --attack-mode continuous --attack-start 1 --attack-clients 0,1
python run.py sim ... --aggregation median          # same attack, robust aggregation

# Regenerate every graph above (62 experiments; uses CUDA if available)
uv run python scripts/plot_attack_benchmark.py --rerun
```

The server logs one `client_update` JSON event per client per round
(`update_norm`, `cosine_to_peers`, `accepted`, `is_attacked`), and attackers
log `attack_injected`.

---

# 9. Member 4 — Detection Baseline

### Main responsibility

Study the model updates produced by clients.

For every client update, record:

```text
client_id
round
update norm
cosine similarity
CKA similarity
```

Initially, CKA is only being studied as a **research signal**.

We are NOT yet implementing the final TRACE-FL trust mechanism.

### Baseline aggregators

Implement/reproduce:

```text
FedAvg
Krum
Median
Trimmed Mean
```

The goal is to establish how existing aggregation behaves under attacks.

### Deliverables

Initial plots:

```text
Update norm:
honest vs malicious

Cosine similarity:
honest vs malicious

CKA:
honest vs malicious
```

---

# 10. Complete Phase 1 Flow

```text
                 DATASET
                    │
                    ▼
             Partition Dataset
                    │
          ┌─────────┴─────────┐
          │                   │
        IID                 Non-IID
          │                   │
          └─────────┬─────────┘
                    ▼
             Flower Clients
                    │
          ┌─────────┴─────────┐
          │                   │
       Honest              Attacker
       Client                Client
          │                   │
          │              AttackEngine
          │                   │
          └─────────┬─────────┘
                    ▼
              Local Training
                    │
                    ▼
             Client Updates
                    │
                    ▼
               ServerApp
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        Norm      Cosine      CKA
          │         │         │
          └─────────┼─────────┘
                    ▼
            Baseline Analysis
                    │
                    ▼
          FedAvg / Krum / Median
                    │
                    ▼
             Experimental Results
```

---

# 11. Phase 1 Experiments

Use initially:

```text
20 clients
20 communication rounds
```

Then vary:

```text
IID
Dirichlet α = 1.0
Dirichlet α = 0.5
Dirichlet α = 0.1
```

### Attack percentage

Start with:

```text
10% malicious
20% malicious
```

Later expand.

---

# 12. Metrics

Record:

### Model performance

```text
Global accuracy
Training loss
Round-to-round convergence
```

### Attack impact

```text
Accuracy degradation
Global model degradation
```

### Update behavior

```text
Update norm
Cosine similarity
CKA similarity
```

### Communication

```text
Model bytes downloaded
Model bytes uploaded
Total bytes per round
Total bytes across experiment
```

---

# 13. Phase 1 Main Research Question

The most important experiment is:

> **Under non-IID data, can representation-level similarity such as CKA distinguish malicious updates from legitimate client heterogeneity?**

We specifically want to avoid assuming:

```text
different update = malicious
```

because non-IID clients naturally produce different updates.

Instead we want to observe:

```text
                   CLIENT UPDATES

              ┌────────────────────┐
              │                    │
         Honest non-IID        Malicious
              │                    │
              ▼                    ▼
        Natural drift      Poisoned drift
              │                    │
              └────────┬───────────┘
                       ▼
                 CKA / Norm /
                 Cosine Analysis
                       │
                       ▼
             Is there separation?
```

If CKA gives useful separation, it becomes the foundation for the next phase.

If it does not, we investigate why and add complementary signals.

---

# 14. What We Are NOT Building in Phase 1

Do NOT implement these yet:

```text
✗ Temporal trust
✗ Authentication challenge-response
✗ PKI
✗ Sybil detection
✗ Collusion graph
✗ Adaptive attacker
✗ Feature freezing
✗ Virtual aggregation
✗ Communication scheduler
✗ TRACE-FL final trust engine
✗ TRACE-FL-NIDS
✗ Production cloud deployment
```

These belong to later phases.

---

# 15. Phase 1 Definition of Done

Phase 1 is complete when the team can run:

```text
20 Flower clients
        ↓
MNIST / Fashion-MNIST
        ↓
IID / non-IID
        ↓
FedAvg
        ↓
attack selected clients
        ↓
collect client updates
        ↓
compute norm / cosine / CKA
        ↓
compare honest vs malicious behavior
        ↓
run baseline aggregators
        ↓
produce reproducible plots
```

And all four members' modules integrate into the same pipeline.

---

# 16. Repository Structure

```text
trace-fl/
│
├── client/
│   ├── client_app.py
│   ├── trainer.py
│   └── model.py
│
├── server/
│   ├── server_app.py
│   └── strategies.py
│
├── datasets/
│   ├── mnist.py
│   ├── fashion_mnist.py
│   └── partition.py
│
├── attacks/
│   ├── label_flip.py
│   ├── gaussian.py
│   ├── sign_flip.py
│   ├── scaling.py
│   └── attack_engine.py
│
├── detection/
│   ├── norms.py
│   ├── cosine.py
│   └── cka.py
│
├── aggregation/
│   ├── fedavg.py
│   ├── krum.py
│   ├── median.py
│   └── trimmed_mean.py
│
├── experiments/
│   ├── configs/
│   └── results/
│
├── plots/
│
├── tests/
│
├── pyproject.toml
└── README.md
```

---

# 17. What comes immediately after Phase 1

Once Phase 1 produces reliable evidence:

```text
Phase 1
FL + attacks + CKA evidence
        ↓
Phase 2
Temporal Trust
        ↓
Phase 3
Identity / Authentication
        ↓
Phase 4
Trust-aware Aggregation
        ↓
Phase 5
Communication Optimization
        ↓
Phase 6
Adaptive / Collusion Attacks
        ↓
Phase 7
NIDS Extension
```

That way you're **not overengineering before you know your central detection hypothesis works**.

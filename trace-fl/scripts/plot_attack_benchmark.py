"""
TRACE-FL attack benchmark: measure how much each attack hurts the global model.

Runs FL rounds in-process (no Flower/Ray) but through the *same* components the
Flower app uses: FederatedDatasetProvider, AttackEngine, train/evaluate,
UpdateValidator, the aggregators in federation/strategy.py and
federation/update_stats.py. So a number here is what the ServerApp would see,
minus Flower's transport. (Flower simulation is ~10x slower per run and its logs
are not a convenient source for sweeps; use `run.py sim --attack-type ...` to
confirm a single configuration end to end.)

Suites (each experiment is cached in the JSON; --rerun recomputes):
  rounds     honest + 4 attacks at HEADLINE intensity, 20% attackers, IID and Dirichlet(0.5)
  intensity  final accuracy vs attack intensity, 20% attackers, IID
  fraction   final accuracy vs % malicious clients, HEADLINE intensity, IID
  defense    each attack under FedAvg / FedAvg+norm bound / trimmed mean / median
  targeted   targeted label flip 7 -> 1 (30% attackers): per-class recall

Usage (from trace-fl/):
  uv run python scripts/plot_attack_benchmark.py                 # replot from cached JSON
  uv run python scripts/plot_attack_benchmark.py --rerun         # recompute everything (~15 min, 8 workers)
  uv run python scripts/plot_attack_benchmark.py --suite rounds --rerun
  uv run python scripts/plot_attack_benchmark.py --demo sign_flip  # one quick live run, prints accuracy per round

Outputs: artifacts/benchmarks/attack_benchmark_results.json and *.png
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # FederatedDatasetProvider reads ./data

import numpy as np  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "benchmarks"
RESULTS_JSON = OUT_DIR / "attack_benchmark_results.json"

ATTACKS = ["label_flipping", "gaussian_noise", "sign_flip", "scaling"]

# Intensity used when an experiment does not sweep it: the smallest swept value
# that clearly hurts FedAvg with 20% attackers (from the intensity suite). Label
# flipping has no such value at 20%, so it uses its maximum.
HEADLINE = {
    "label_flipping": 1.0,   # flip 100% of the attacker's labels (y -> 9 - y)
    "gaussian_noise": 300.0, # noise sigma = 300 x per-coordinate RMS of the honest update
    "sign_flip": 4.0,        # send -4 x delta (= critical gamma (1-f)/f for f = 0.2)
    "scaling": 10.0,         # send 10 x delta
}

SWEEP = {
    "label_flipping": [0.25, 0.5, 0.75, 1.0],
    "gaussian_noise": [10.0, 100.0, 300.0, 1000.0],
    "sign_flip": [1.0, 2.0, 4.0, 8.0],
    "scaling": [5.0, 10.0, 50.0, 100.0],
}

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]

COMMON = {
    "dataset": "mnist",
    "partition": "iid",
    "alpha": 0.5,
    "num_clients": 10,
    "rounds": 10,
    "epochs": 1,
    "batch_size": 32,
    "lr": 0.01,
    "seed": 42,
    "fraction": 0.2,
    "aggregation": "fedavg",
    "trim_ratio": 0.2,
    "max_update_norm": None,
    "params": {},
}


# ----------------------------------------------------------------------------
# Experiment execution
# ----------------------------------------------------------------------------

def spec(name: str, **overrides) -> Dict[str, Any]:
    s = dict(COMMON)
    s.update(overrides)
    s["name"] = name
    return s


def run_experiment(s: Dict[str, Any]) -> Dict[str, Any]:
    """Run one FL experiment and return per-round metrics + per-client update stats."""
    import torch

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "4")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set: pool workers are reused across experiments
        pass
    torch.manual_seed(s["seed"])

    from attacks.engine import AttackEngine
    from core.model import create_model, get_parameters, set_parameters
    from data.dataset import FederatedDatasetProvider
    from federation.strategy import build_aggregation_strategy
    from federation.update_stats import compute_update_stats
    from federation.update_validator import UpdateValidator
    from training.trainer import evaluate, train

    # CUDA only changes speed (the project's lockfile is CPU-only torch; a CUDA env
    # can be pointed at this script for big sweeps).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = s["num_clients"]
    provider = FederatedDatasetProvider(
        num_clients=n, batch_size=s["batch_size"], seed=s["seed"],
        dataset_name=s["dataset"], partition_type=s["partition"], alpha=s["alpha"],
    )
    cache_tensors(provider)
    partitions = [provider.get_partition(i) for i in range(n)]
    test_loader = provider.get_global_test_loader()

    attack_type = s.get("attack")
    config = {
        "num_clients": n,
        "seed": s["seed"],
        "attack": {
            "enabled": attack_type is not None,
            "type": attack_type or "sign_flip",
            "intensity": s.get("intensity", 1.0),
            "schedule": {"mode": "continuous", "start_round": 1, "end_round": s["rounds"]},
            "targets": {"mode": "fraction", "fraction": s["fraction"]},
            "params": s["params"],
        },
    }
    engine = AttackEngine.from_config(config)
    malicious = engine.attack.schedule.malicious_indices(n) if engine.enabled else []

    model = create_model()
    aggregator = build_aggregation_strategy({"aggregation": s["aggregation"], "trim_ratio": s["trim_ratio"]})
    validator = UpdateValidator(
        expected_shapes=[p.shape for p in get_parameters(model)],
        max_update_norm=s["max_update_norm"],
    )

    def eval_record(rnd):
        loss, acc, d = evaluate(model, test_loader, device, return_details=True)
        cm = np.array(d["confusion_matrix"])
        k = cm.shape[0]
        # Share of test images predicted as (9 - y): success rate of the permutation flip.
        flip_rate = float(sum(cm[y, k - 1 - y] for y in range(k)) / cm.sum())
        return {"round": rnd, "loss": float(loss), "accuracy": float(acc),
                "macro_f1": d["macro_f1"], "class_recall": d["class_recall"],
                "permutation_flip_rate": flip_rate}

    rounds = [eval_record(0)]
    client_stats: List[Dict[str, Any]] = []

    for rnd in range(1, s["rounds"] + 1):
        global_params = get_parameters(model)
        sent, attacked_flags = [], []
        for cid in range(n):
            cname = f"client_{cid}"
            local = create_model()
            set_parameters(local, global_params)
            loader, pre = engine.apply_pre_training(partitions[cid].train_loader, rnd, cname, n)
            train(local, loader, s["epochs"], s["lr"], device)
            params, post, _ = engine.apply_post_training(global_params, get_parameters(local), rnd, cname, n)
            sent.append((partitions[cid].num_examples, params))
            attacked_flags.append(pre or post)

        checks = [validator.check(ne, p, global_params) for ne, p in sent]
        stats = compute_update_stats(global_params, [p for _, p in sent])
        for cid, ((ne, _), (ok, why), st) in enumerate(zip(sent, checks, stats)):
            client_stats.append({"round": rnd, "client": cid, "malicious": attacked_flags[cid],
                                 "num_examples": ne, "accepted": ok, "reason": why, **st})

        accepted = [u for u, (ok, _) in zip(sent, checks) if ok]
        if accepted:  # otherwise keep the previous global model, like the ServerApp
            set_parameters(model, [a.astype(np.float32) for a in aggregator.aggregate(accepted)])
        rounds.append(eval_record(rnd))
        print(f"    [{s['name']}] round {rnd}/{s['rounds']} acc {rounds[-1]['accuracy'] * 100:.1f}%", flush=True)

    total = sum(p.num_examples for p in partitions)
    return {
        "spec": s,
        "malicious_clients": malicious,
        "malicious_data_share": sum(partitions[i].num_examples for i in malicious) / total,
        "rounds": rounds,
        "client_stats": client_stats,
        "final_accuracy": rounds[-1]["accuracy"],
    }


class _CachedImages:
    """Same (image, label) pairs as the torchvision dataset, transformed once.

    ToTensor + Normalize are deterministic, so this only removes the per-sample
    PIL round trip; indices, partitions and DataLoader order are unchanged.
    """

    def __init__(self, ds):
        norm = ds.transform.transforms[-1]
        x = ds.data.float().div(255.0).unsqueeze(1)
        self.x = (x - norm.mean[0]) / norm.std[0]
        self.targets = ds.targets
        self.classes = ds.classes
        self._labels = ds.targets.tolist()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        return self.x[i], self._labels[i]


def cache_tensors(provider) -> None:
    provider.train_dataset = _CachedImages(provider.train_dataset)
    provider.test_dataset = _CachedImages(provider.test_dataset)
    provider._partition_datasets()  # same seeds -> same partitions over the cached data


def save_result(name: str, res: Dict[str, Any]) -> None:
    """Read-modify-write so concurrent benchmark processes don't drop each other's results."""
    current = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    current[name] = res
    tmp = RESULTS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(current))
    tmp.replace(RESULTS_JSON)


def build_suite(suite: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = []
    if suite == "rounds":
        for part in ("iid", "dirichlet"):
            specs.append(spec(f"rounds/{part}/honest", partition=part))
            for a in ATTACKS:
                specs.append(spec(f"rounds/{part}/{a}", partition=part, attack=a, intensity=HEADLINE[a]))
    elif suite == "intensity":
        for a in ATTACKS:
            for v in SWEEP[a]:
                specs.append(spec(f"intensity/{a}/{v:g}", attack=a, intensity=v))
    elif suite == "fraction":
        for a in ATTACKS:
            for f in FRACTIONS:
                specs.append(spec(f"fraction/{a}/{f:g}", attack=a, intensity=HEADLINE[a], fraction=f))
    elif suite == "defense":
        bound = norm_bound_from_honest(results)
        for agg, extra in (("fedavg_normbound", {"max_update_norm": bound}),
                           ("trimmed_mean", {"aggregation": "trimmed_mean"}),
                           ("median", {"aggregation": "median"})):
            specs.append(spec(f"defense/{agg}/honest", **extra))
            for a in ATTACKS:
                specs.append(spec(f"defense/{agg}/{a}", attack=a, intensity=HEADLINE[a], **extra))
    elif suite == "targeted":
        specs.append(spec("targeted/label_flip_7to1", attack="label_flipping", intensity=1.0, fraction=0.3,
                          params={"mode": "targeted", "source_class": 7, "target_class": 1}))
    return specs


def norm_bound_from_honest(results: Dict[str, Any]) -> float:
    """1.5 x the largest honest update norm seen in the honest IID run.

    A fixed bound like this is what a server could calibrate on clean rounds.
    It is easy to evade (stay just under it); the plot shows that limitation.
    """
    honest = results.get("rounds/iid/honest")
    if honest is None:
        honest = run_experiment(spec("rounds/iid/honest"))
        results["rounds/iid/honest"] = honest
    return 1.5 * max(c["update_norm"] for c in honest["client_stats"])


SUITES = ["rounds", "intensity", "fraction", "defense", "targeted"]


def run_suites(suites: List[str], rerun: bool, workers: int) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
        if "honest" in results:  # old format from the previous script
            results = {}

    # Defense needs the honest baseline (for its norm bound), so it runs second.
    stages = [[x for x in suites if x != "defense"], [x for x in suites if x == "defense"]]
    for stage in stages:
        todo = []
        for suite in stage:
            for s in build_suite(suite, results):
                if rerun or s["name"] not in results:
                    todo.append(s)
        todo = list({s["name"]: s for s in todo}.values())
        if not todo:
            continue
        print(f"[benchmark] running {len(todo)} experiments with {workers} workers ...")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for s, res in zip(todo, pool.map(run_experiment, todo)):
                results[s["name"]] = res
                print(f"  {s['name']:<40} final acc {res['final_accuracy'] * 100:6.2f}%", flush=True)
                save_result(s["name"], res)
        print(f"[benchmark] stage done in {time.time() - t0:.0f}s -> {RESULTS_JSON.relative_to(ROOT)}")
    # Re-read: another benchmark process may have added results meanwhile.
    return json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else results


# ----------------------------------------------------------------------------
# Plotting (palette validated with the dataviz validator: all-pairs, light mode)
# ----------------------------------------------------------------------------

LABELS = {"honest": "No attack", "label_flipping": "Label flipping", "gaussian_noise": "Gaussian noise",
          "sign_flip": "Sign flip", "scaling": "Scaling / boosting"}
COLORS = {"label_flipping": "#2a78d6", "gaussian_noise": "#eb6834", "sign_flip": "#1baf7a",
          "scaling": "#4a3aa7", "honest": "#52514e"}
MARKERS = {"label_flipping": "o", "gaussian_noise": "s", "sign_flip": "^", "scaling": "D", "honest": "o"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=12, color=INK, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    ax.set_ylabel(ylabel, fontsize=10, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def _spread(ys, min_gap):
    """Nudge label y-positions apart (keeps order) so end labels don't collide."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def _end_labels(ax, points, min_gap, fmt="{:.1f}%"):
    """points: list of (x, y, text_or_None). Draws labels right of the last point."""
    ys = _spread([p[1] for p in points], min_gap)
    for (x, y, text), ly in zip(points, ys):
        ax.annotate(text if text is not None else fmt.format(y), (x, y), xytext=(8, 0),
                    textcoords="offset points", fontsize=8.5, color=INK, va="center",
                    annotation_clip=False)
        if ly != y:  # move the text, keep the anchor
            ax.texts[-1].set_position((8, (ly - y) * ax.bbox.height / (ax.get_ylim()[1] - ax.get_ylim()[0])
                                       * 72 / ax.figure.dpi))


def _plain_log_ticks(axis):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    # 1-2-5 steps so a panel spanning less than a decade still gets labels.
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _subtitle(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8.5, color=INK2, ha="left", va="bottom")


def plot_rounds(results, plt):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, part, title in zip(axes, ("iid", "dirichlet"), ("IID partition", "Non-IID, Dirichlet α = 0.5")):
        _style(ax, f"Global test accuracy per round — {title}", "Federated round", "Accuracy (%)")
        ax.set_ylim(0, 102)
        ax.set_xlim(-0.3, COMMON["rounds"] + 1.4)
        ends = []
        for key in ["honest"] + ATTACKS:
            r = results.get(f"rounds/{part}/{key}")
            if not r:
                continue
            xs = [m["round"] for m in r["rounds"]]
            ys = [m["accuracy"] * 100 for m in r["rounds"]]
            label = LABELS[key] + ("" if key == "honest" else f" ({HEADLINE[key]:g})")
            ax.plot(xs, ys, color=COLORS[key], marker=MARKERS[key], markersize=6, linewidth=2,
                    linestyle="--" if key == "honest" else "-", label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ends.append((xs[-1], ys[-1], None))
        _end_labels(ax, ends, min_gap=4.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.04))
    _subtitle(fig, "MNIST, 10 clients, 2 malicious (20%), FedAvg, 1 local epoch, SGD lr 0.01. "
                   "Intensity in brackets. Round 0 = untrained model.")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUT_DIR / "attack_accuracy_vs_round.png", facecolor=SURFACE)
    plt.close(fig)


def plot_intensity(results, plt):
    honest = results.get("rounds/iid/honest")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    xlab = {"label_flipping": "Fraction of attacker labels flipped", "gaussian_noise": "Noise σ (× update RMS)",
            "sign_flip": "γ (sends −γ·Δ)", "scaling": "β (sends β·Δ)"}
    for ax, a in zip(axes, ATTACKS):
        xs = [v for v in SWEEP[a] if f"intensity/{a}/{v:g}" in results]
        ys = [results[f"intensity/{a}/{v:g}"]["final_accuracy"] * 100 for v in xs]
        if honest:
            ax.axhline(honest["final_accuracy"] * 100, color=COLORS["honest"], linestyle="--", linewidth=1.2)
            ax.text(xs[0] if xs else 0, honest["final_accuracy"] * 100 + 2, "no attack", fontsize=8, color=INK2)
        ax.plot(xs, ys, color=COLORS[a], marker=MARKERS[a], markersize=7, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        for i, (x, y) in enumerate(zip(xs, ys)):
            drops_next = i + 1 < len(ys) and ys[i + 1] < y - 20
            if y < 50:      # low point: label above
                off, ha = (0, 9), "center"
            elif drops_next:  # line falls to the right: label to the left
                off, ha = (-7, -12), "right"
            else:
                off, ha = (0, -15), "center"
            ax.annotate(f"{y:.1f}", (x, y), xytext=off, textcoords="offset points",
                        fontsize=8, color=INK, ha=ha)
        if a != "label_flipping":
            ax.set_xscale("log")
            _plain_log_ticks(ax.xaxis)
            ax.set_xticks(xs)
        _style(ax, LABELS[a], xlab[a], "Final accuracy (%)" if a == ATTACKS[0] else "")
        ax.set_ylim(0, 105)
    _subtitle(fig, "Accuracy after 10 rounds. MNIST IID, 10 clients, 2 malicious (20%), FedAvg.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "attack_intensity_sweep.png", facecolor=SURFACE)
    plt.close(fig)


def plot_fraction(results, plt):
    honest = results.get("rounds/iid/honest")
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    if honest:
        ax.axhline(honest["final_accuracy"] * 100, color=COLORS["honest"], linestyle="--", linewidth=1.2,
                   label="No attack")
    ends = []
    for a in ATTACKS:
        xs = [f for f in FRACTIONS if f"fraction/{a}/{f:g}" in results]
        ys = [results[f"fraction/{a}/{f:g}"]["final_accuracy"] * 100 for f in xs]
        if not xs:
            continue
        ax.plot([x * 100 for x in xs], ys, color=COLORS[a], marker=MARKERS[a], markersize=7, linewidth=2,
                label=f"{LABELS[a]} ({HEADLINE[a]:g})", markeredgecolor=SURFACE, markeredgewidth=1.2)
        ends.append((xs[-1] * 100, ys[-1], LABELS[a]))
    _style(ax, "Final accuracy vs share of malicious clients", "Malicious clients (%)", "Final accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(5, 64)
    _end_labels(ax, ends, min_gap=5)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    _subtitle(fig, "MNIST IID, 10 clients, FedAvg, 10 rounds. Attack intensity in brackets.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "attack_fraction_sweep.png", facecolor=SURFACE)
    plt.close(fig)


def plot_signatures(results, plt):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for row, part in enumerate(("iid", "dirichlet")):
        for col, a in enumerate(ATTACKS):
            ax = axes[row, col]
            r = results.get(f"rounds/{part}/{a}")
            if not r:
                continue
            hon = [c for c in r["client_stats"] if not c["malicious"]]
            mal = [c for c in r["client_stats"] if c["malicious"]]
            ax.scatter([c["cosine_to_peers"] for c in hon], [c["update_norm"] for c in hon], s=28,
                       color="#8a8984", edgecolor=SURFACE, linewidth=0.8, zorder=2)
            ax.scatter([c["cosine_to_peers"] for c in mal], [c["update_norm"] for c in mal], s=42,
                       color=COLORS[a], marker=MARKERS[a], edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.set_yscale("log")
            _plain_log_ticks(ax.yaxis)
            ax.axvline(0, color=GRID, linewidth=1)
            title = f"{LABELS[a].split(' /')[0]} — {'IID' if part == 'iid' else 'Dirichlet 0.5'}"
            _style(ax, title, "Cosine to peers' median update" if row == 1 else "",
                   "Update L2 norm (log)" if col == 0 else "")
            ax.title.set_fontsize(10.5)
            ax.text(0.02, 0.02, f"{len(hon)} honest · {len(mal)} malicious updates (overlapping points stack)",
                    transform=ax.transAxes, fontsize=7.5, color=INK2)
    handles = [Line2D([], [], marker="o", linestyle="", color="#8a8984", markersize=7, label="Honest client")]
    handles += [Line2D([], [], marker=MARKERS[a], linestyle="", color=COLORS[a], markersize=7,
                       label=f"Malicious: {LABELS[a]}") for a in ATTACKS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.03))
    _subtitle(fig, "Every client update from rounds 1–10 (headline intensities, 2 of 10 clients malicious). "
                   "Cosine is leave-one-out: each update vs the coordinate-wise median of the other nine.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_DIR / "attack_update_signatures.png", facecolor=SURFACE)
    plt.close(fig)


def plot_defense(results, plt):
    rows = ["honest"] + ATTACKS
    cols = [("rounds/iid/{}", "FedAvg"), ("defense/fedavg_normbound/{}", "FedAvg +\nnorm bound"),
            ("defense/trimmed_mean/{}", "Trimmed mean\n(20%)"), ("defense/median/{}", "Coordinate\nmedian")]
    grid = np.full((len(rows), len(cols)), np.nan)
    for i, a in enumerate(rows):
        for j, (pat, _) in enumerate(cols):
            r = results.get(pat.format(a))
            if r:
                grid[i, j] = r["final_accuracy"] * 100
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}%", ha="center", va="center", fontsize=11,
                        color="#ffffff" if grid[i, j] > 60 else INK, fontweight="semibold")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r] + ("" if r == "honest" else f" ({HEADLINE[r]:g})") for r in rows],
                       fontsize=9, color=INK2)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Final accuracy by aggregation rule (darker = better)", fontsize=12, color=INK,
                 loc="left", fontweight="semibold")
    _subtitle(fig, "MNIST IID, 10 clients, 2 malicious, 10 rounds. Norm bound = 1.5 × largest honest "
                   "update norm in the clean run.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "attack_defense_matrix.png", facecolor=SURFACE)
    plt.close(fig)


def plot_targeted(results, plt):
    r = results.get("targeted/label_flip_7to1")
    honest = results.get("rounds/iid/honest")
    if not (r and honest):
        return
    classes = list(range(10))
    h = [honest["rounds"][-1]["class_recall"][str(c)] * 100 for c in classes]
    t = [r["rounds"][-1]["class_recall"][str(c)] * 100 for c in classes]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(10)
    w = 0.38
    ax.bar(x - w / 2 - 0.01, h, w, color="#8a8984", label="No attack")
    ax.bar(x + w / 2 + 0.01, t, w, color=COLORS["label_flipping"], label="Targeted flip 7→1 (30% clients)")
    ax.annotate(f"{t[7]:.1f}%", (7 + w / 2, t[7]), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
    ax.annotate(f"{h[7]:.1f}%", (7 - w / 2, h[7]), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
    _style(ax, "Per-class recall: targeted label flip hides inside global accuracy",
           "Digit class", "Recall (%)")
    ax.set_xticks(x)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    _subtitle(fig, f"Global accuracy: {honest['final_accuracy'] * 100:.1f}% without attack vs "
                   f"{r['final_accuracy'] * 100:.1f}% with it. MNIST IID, 10 clients, FedAvg, 10 rounds.")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_DIR / "attack_targeted_label_flip.png", facecolor=SURFACE)
    plt.close(fig)


def plot_all(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fn in (plot_rounds, plot_intensity, plot_fraction, plot_signatures, plot_defense, plot_targeted):
        fn(results, plt)
    print(f"[benchmark] plots written to {OUT_DIR.relative_to(ROOT)}/")


def print_summary(results):
    honest = results.get("rounds/iid/honest")
    if not honest:
        return
    base = honest["final_accuracy"] * 100
    print(f"\n{'experiment':<42}{'final acc':>10}{'drop':>9}")
    for name in sorted(results):
        acc = results[name]["final_accuracy"] * 100
        print(f"{name:<42}{acc:>9.2f}%{acc - base:>8.2f}")


def demo(attack: str, rounds: int):
    """Small, fast run for a live demo: 5 clients, 1 malicious, prints each round."""
    for name, a in (("no attack", None), (attack, attack)):
        s = spec(f"demo/{name}", num_clients=5, rounds=rounds, attack=a,
                 intensity=HEADLINE.get(attack, 1.0), fraction=0.2)
        r = run_experiment(s)
        accs = " ".join(f"{m['accuracy'] * 100:5.1f}" for m in r["rounds"][1:])
        mal = [c for c in r["client_stats"] if c["malicious"]]
        print(f"{name:<16} acc/round: {accs}")
        if mal:
            hon = [c for c in r["client_stats"] if not c["malicious"]]
            print(f"{'':<16} malicious cos={np.mean([c['cosine_to_peers'] for c in mal]):+.2f} "
                  f"norm={np.mean([c['update_norm'] for c in mal]):.2f} | honest "
                  f"cos={np.mean([c['cosine_to_peers'] for c in hon]):+.2f} "
                  f"norm={np.mean([c['update_norm'] for c in hon]):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES + ["all"], default="all")
    ap.add_argument("--rerun", action="store_true", help="recompute even if cached")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 4)),
                    help="parallel experiments; each uses BENCH_THREADS (default 4) torch threads")
    ap.add_argument("--plot-only", action="store_true", help="only redraw from cached JSON")
    ap.add_argument("--demo", choices=ATTACKS, help="quick 5-client run of one attack vs no attack")
    ap.add_argument("--demo-rounds", type=int, default=3)
    args = ap.parse_args()

    if args.demo:
        demo(args.demo, args.demo_rounds)
        return

    if args.plot_only:
        results = json.loads(RESULTS_JSON.read_text())
    else:
        suites = SUITES if args.suite == "all" else [args.suite]
        if "defense" in suites and "rounds" not in suites:  # needs the honest baseline
            suites = ["rounds"] + suites
        results = run_suites(suites, args.rerun, args.workers)
    plot_all(results)
    print_summary(results)


if __name__ == "__main__":
    main()

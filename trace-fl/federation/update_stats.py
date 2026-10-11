"""Per-client update statistics (measurement only; no client is rejected here).

These are the Phase 1 "does the malicious update look different?" signals:
  update_norm       ||w_client - w_global||_2
  cosine_to_peers   cosine(delta_client, coordinate-wise median of the *other*
                    clients' deltas)

Why leave-one-out median: a reference that includes the client itself biases
its cosine toward +1, and a *mean* reference is dragged by the attacker (one
client sending -4*delta made every honest client look anti-aligned, cos ~ -0.9).
"""

from typing import Dict, List

import numpy as np

from core.types import NDArrays


def _flat_delta(parameters: NDArrays, global_parameters: NDArrays) -> np.ndarray:
    return np.concatenate([
        (p.astype(np.float64) - g).ravel() for p, g in zip(parameters, global_parameters)
    ])


def compute_update_stats(
    global_parameters: NDArrays,
    client_parameters: List[NDArrays],
) -> List[Dict[str, float]]:
    """Return one {update_norm, cosine_to_peers} dict per client, in input order."""
    if not client_parameters:
        return []
    deltas = np.stack([_flat_delta(p, global_parameters) for p in client_parameters])
    norms = np.linalg.norm(deltas, axis=1)
    n = len(client_parameters)
    stats = []
    for i in range(n):
        if n > 1:
            peers = np.median(np.delete(deltas, i, axis=0), axis=0)
            denom = max(norms[i], 1e-12) * max(np.linalg.norm(peers), 1e-12)
            cos = float(np.dot(deltas[i], peers) / denom)
        else:
            cos = 1.0
        stats.append({"update_norm": float(norms[i]), "cosine_to_peers": cos})
    return stats

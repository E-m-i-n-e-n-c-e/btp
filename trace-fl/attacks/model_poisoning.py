"""
Model and update poisoning attacks for TRACE-FL.

Includes:
1. GaussianNoiseAttack: Stochastic noise injection into weight updates.
2. SignFlipAttack: Mathematical gradient/update reversal (-gamma * delta_w).
3. ScalingAttack: Update boosting/amplification (beta * delta_w) to dominate FedAvg.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from attacks.base import AttackSchedule, ModelPoisoningAttack, parse_client_index


def compute_vector_metrics(
    honest_delta: List[np.ndarray],
    poisoned_delta: List[np.ndarray],
) -> Dict[str, float]:
    """Compute L2 norms and cosine similarity between honest and poisoned updates."""
    flat_h = np.concatenate([d.flatten() for d in honest_delta])
    flat_p = np.concatenate([d.flatten() for d in poisoned_delta])

    norm_h = float(np.linalg.norm(flat_h))
    norm_p = float(np.linalg.norm(flat_p))

    eps = 1e-12
    cos_sim = float(
        np.dot(flat_h, flat_p) / (max(norm_h, eps) * max(norm_p, eps))
    )
    perturbation_norm = float(np.linalg.norm(flat_p - flat_h))

    return {
        "l2_norm_honest": norm_h,
        "l2_norm_poisoned": norm_p,
        "cosine_similarity": cos_sim,
        "perturbation_norm": perturbation_norm,
    }


class GaussianNoiseAttack(ModelPoisoningAttack):
    """
    Gaussian Noise Attack.

    Adds Gaussian noise to the model update delta:
        delta_w = w_local - w_global
        delta_poisoned = delta_w + N(0, sigma^2)
        w_poisoned = w_global + delta_poisoned
    """

    def __init__(
        self,
        schedule: AttackSchedule,
        intensity: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name="gaussian_noise",
            schedule=schedule,
            intensity=intensity,
            params=params,
        )
        self.scale_by_norm = self.params.get("scale_by_norm", True)
        if self.intensity < 0:
            raise ValueError("gaussian_noise intensity (sigma multiplier) must be >= 0.")

    def poison_update(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        round_num: int,
        client_id: Union[str, int] = 0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        # Seed per (round, client): without the client term every attacker in a
        # round drew the *same* noise vector, which then does not average out.
        cid_idx = parse_client_index(client_id) or 0
        rng = np.random.RandomState(self.schedule.seed + round_num * 100 + cid_idx * 7919)

        # 1. Compute delta
        deltas = [
            loc - glob
            for loc, glob in zip(local_parameters, global_parameters)
        ]

        total_dim = sum(d.size for d in deltas)
        flat_delta = np.concatenate([d.flatten() for d in deltas])
        delta_norm = float(np.linalg.norm(flat_delta))

        # Determine standard deviation
        if self.scale_by_norm:
            sigma = self.intensity * (delta_norm / np.sqrt(max(total_dim, 1)))
        else:
            sigma = float(self.intensity)

        # 2. Inject noise into delta
        poisoned_deltas = []
        for d in deltas:
            noise = rng.normal(loc=0.0, scale=sigma, size=d.shape).astype(
                d.dtype
            )
            poisoned_deltas.append(d + noise)

        # 3. Compute final poisoned parameters
        poisoned_params = [
            np.ascontiguousarray(
                (glob + p_d).astype(np.float32)
            )
            for glob, p_d in zip(global_parameters, poisoned_deltas)
        ]

        metrics = compute_vector_metrics(deltas, poisoned_deltas)
        metrics["noise_sigma"] = float(sigma)

        return poisoned_params, metrics


class SignFlipAttack(ModelPoisoningAttack):
    """
    Sign Flip Attack (Gradient Reversal).

    Reverses the direction of the local update delta:
        delta_w = w_local - w_global
        delta_poisoned = -gamma * delta_w
        w_poisoned = w_global + delta_poisoned = w_global - gamma * (w_local - w_global)
    """

    def __init__(
        self,
        schedule: AttackSchedule,
        intensity: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name="sign_flip",
            schedule=schedule,
            intensity=intensity,
            params=params,
        )

    def poison_update(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        round_num: int,
        client_id: Union[str, int] = 0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        gamma = float(self.intensity)

        # 1. Compute delta
        deltas = [
            loc - glob
            for loc, glob in zip(local_parameters, global_parameters)
        ]

        # 2. Invert delta
        poisoned_deltas = [-gamma * d for d in deltas]

        # 3. Compute final poisoned parameters
        poisoned_params = [
            np.ascontiguousarray(
                (glob + p_d).astype(np.float32)
            )
            for glob, p_d in zip(global_parameters, poisoned_deltas)
        ]

        metrics = compute_vector_metrics(deltas, poisoned_deltas)
        metrics["gamma"] = gamma

        return poisoned_params, metrics


class ScalingAttack(ModelPoisoningAttack):
    """
    Scaling / Boosting Attack.

    Amplifies the local update delta by a large factor beta:
        delta_w = w_local - w_global
        delta_poisoned = beta * delta_w
        w_poisoned = w_global + beta * delta_w
    """

    def __init__(
        self,
        schedule: AttackSchedule,
        intensity: float = 10.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name="scaling",
            schedule=schedule,
            intensity=intensity,
            params=params,
        )
        # params.boost_factor wins over intensity (kept for backwards compatibility).
        self.boost_factor = float(
            self.params.get("boost_factor", self.intensity)
        )
        if self.boost_factor == 1.0:
            # beta = 1 sends the honest update unchanged; usually a config mistake
            # (config.yaml's default intensity is 1.0).
            warnings.warn("scaling attack with boost_factor=1.0 is a no-op; set attack.intensity (e.g. 10).")

    def poison_update(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        round_num: int,
        client_id: Union[str, int] = 0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        beta = self.boost_factor

        # 1. Compute delta
        deltas = [
            loc - glob
            for loc, glob in zip(local_parameters, global_parameters)
        ]

        # 2. Scale delta
        poisoned_deltas = [beta * d for d in deltas]

        # 3. Compute final poisoned parameters
        poisoned_params = [
            np.ascontiguousarray(
                (glob + p_d).astype(np.float32)
            )
            for glob, p_d in zip(global_parameters, poisoned_deltas)
        ]

        metrics = compute_vector_metrics(deltas, poisoned_deltas)
        metrics["boost_factor"] = beta

        return poisoned_params, metrics

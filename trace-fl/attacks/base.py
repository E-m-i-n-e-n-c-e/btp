"""
Base classes and schedule definitions for the TRACE-FL Attack Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader


VALID_SCHEDULE_MODES = ("continuous", "intermittent", "custom_rounds", "probabilistic")
VALID_TARGET_MODES = ("client_ids", "fraction")


def parse_client_index(client_id: Union[str, int]) -> Optional[int]:
    """Map 3, "3" or "client_3" to 3; anything else to None (never targeted)."""
    if isinstance(client_id, int):
        return client_id
    tail = str(client_id).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


@dataclass
class AttackSchedule:
    """
    Defines when and where an attack is active.

    Attributes:
        mode: Scheduling strategy:
            - 'continuous': Active every round in [start_round, end_round].
            - 'intermittent': Active on cadence (round - start_round) % interval == 0.
            - 'custom_rounds': Active only on explicit list of rounds.
            - 'probabilistic': Active with probability p in [start_round, end_round].
        start_round: First round where attack is eligible.
        end_round: Last round of eligibility.
        interval: Cadence interval for intermittent mode (e.g. 2 -> every other round).
        rounds: Specific list of rounds for 'custom_rounds' mode.
        probability: Probability for 'probabilistic' mode.
        client_ids: Explicit list of client IDs to attack (e.g. ['client_0', 'client_2'] or [0, 2]).
        fraction: Fraction of total clients to attack; the first round(N * fraction)
            client indices (min 1) are malicious.
        target_mode: 'client_ids' | 'fraction' | None. Selects which of the two
            fields above is used when both are present (e.g. a config that keeps
            client_ids while a CLI override sets a fraction). None = infer:
            client_ids if set, else fraction, else client 0.
    """

    mode: str = "continuous"
    start_round: int = 1
    end_round: int = 100
    interval: int = 2
    rounds: List[int] = field(default_factory=list)
    probability: float = 0.5
    client_ids: Optional[List[Union[str, int]]] = None
    fraction: Optional[float] = None
    seed: int = 42
    target_mode: Optional[str] = None

    def __post_init__(self):
        # Fail at config-load time rather than mid-run (e.g. interval=0 used to
        # raise ZeroDivisionError inside a Ray actor on the first attack round).
        if self.mode not in VALID_SCHEDULE_MODES:
            raise ValueError(f"Unknown schedule mode '{self.mode}'. Choose from {VALID_SCHEDULE_MODES}.")
        if self.start_round > self.end_round:
            raise ValueError(f"start_round ({self.start_round}) > end_round ({self.end_round}).")
        if self.interval < 1:
            raise ValueError("interval must be >= 1.")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        if self.fraction is not None and not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        if self.target_mode is not None and self.target_mode not in VALID_TARGET_MODES:
            raise ValueError(f"Unknown targets.mode '{self.target_mode}'. Choose from {VALID_TARGET_MODES}.")
        if self.target_mode == "fraction" and self.fraction is None:
            raise ValueError("targets.mode is 'fraction' but targets.fraction is not set.")
        if self.target_mode == "client_ids" and not self.client_ids:
            raise ValueError("targets.mode is 'client_ids' but targets.client_ids is empty.")

    def malicious_indices(self, num_clients: int) -> List[int]:
        """Client indices this schedule ever targets (useful for logs and plots)."""
        return [i for i in range(num_clients) if self.is_client_targeted(i, num_clients)]

    def is_client_targeted(
        self, client_id: Union[str, int], num_clients: int
    ) -> bool:
        """Determine if a given client is malicious."""
        cid_idx = parse_client_index(client_id)
        if cid_idx is None:
            return False

        use_ids = self.target_mode == "client_ids" or (
            self.target_mode is None and self.client_ids is not None
        )
        use_fraction = self.target_mode == "fraction" or (
            self.target_mode is None and self.client_ids is None and self.fraction is not None
        )

        if use_ids:
            return cid_idx in {parse_client_index(c) for c in self.client_ids}

        if use_fraction:
            num_malicious = max(1, int(round(num_clients * self.fraction)))
            return cid_idx < num_malicious

        return cid_idx == 0

    def is_active(
        self, round_num: int, client_id: Union[str, int], num_clients: int
    ) -> bool:
        """Determine if the attack should execute for client on round_num."""
        if not self.is_client_targeted(client_id, num_clients):
            return False

        if round_num < self.start_round or round_num > self.end_round:
            return False

        if self.mode == "continuous":
            return True

        elif self.mode == "intermittent":
            step = round_num - self.start_round
            return (step % self.interval) == 0

        elif self.mode == "custom_rounds":
            return round_num in self.rounds

        elif self.mode == "probabilistic":
            # Seeded per (round, client) so every process reaches the same decision.
            cid_idx = parse_client_index(client_id) or 0
            rng = random.Random(self.seed + round_num * 1000 + cid_idx)
            return rng.random() < self.probability

        return False


class BaseAttack(ABC):
    """Abstract base class for all FL attacks."""

    def __init__(
        self,
        name: str,
        schedule: AttackSchedule,
        intensity: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.schedule = schedule
        self.intensity = intensity
        self.params = params or {}

    def should_attack(
        self, round_num: int, client_id: Union[str, int], num_clients: int
    ) -> bool:
        """Check if attack conditions are met."""
        return self.schedule.is_active(round_num, client_id, num_clients)


class DataPoisoningAttack(BaseAttack):
    """Base class for data-poisoning attacks (applied to DataLoader)."""

    @abstractmethod
    def poison_train_loader(
        self, train_loader: DataLoader, round_num: int
    ) -> DataLoader:
        """Wrap or transform client DataLoader without mutating underlying memory."""
        pass


class ModelPoisoningAttack(BaseAttack):
    """Base class for model/update-poisoning attacks (applied to weights)."""

    @abstractmethod
    def poison_update(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        round_num: int,
        client_id: Union[str, int] = 0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Perturb local update relative to global model.

        Returns:
            Tuple of (poisoned_parameters, telemetry_metrics_dict)
        """
        pass

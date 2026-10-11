from typing import List, Tuple
import numpy as np
from core.types import NDArrays


class AggregationStrategy:
    """Minimal interface for aggregation strategies."""
    def aggregate(self, updates: List[Tuple[int, NDArrays]]) -> NDArrays:
        """
        Aggregates a list of client updates.
        Each update is a tuple of (num_examples, parameters).
        """
        raise NotImplementedError


class FedAvgStrategy(AggregationStrategy):
    """Federated Averaging strategy implementation.

    Breakdown point 0: a single client can move the average arbitrarily far.
    """

    def aggregate(self, updates: List[Tuple[int, NDArrays]]) -> NDArrays:
        if not updates:
            raise ValueError("No updates provided for aggregation.")

        total_examples = sum(num_examples for num_examples, _ in updates)
        if total_examples == 0:
            raise ValueError("Total number of examples is 0.")

        # Create a list of zero arrays with the same shape as the parameters
        # We assume all updates have the same shape
        first_params = updates[0][1]
        aggregated_params = [np.zeros_like(p) for p in first_params]

        for num_examples, parameters in updates:
            weight = num_examples / total_examples
            for i, p in enumerate(parameters):
                aggregated_params[i] += p * weight

        return aggregated_params


class MedianStrategy(AggregationStrategy):
    """Coordinate-wise median (Yin et al., 2018). Ignores num_examples, so it is
    also immune to inflated sample counts. Tolerates < 50% malicious clients per
    coordinate, at some accuracy cost under non-IID data."""

    def aggregate(self, updates: List[Tuple[int, NDArrays]]) -> NDArrays:
        if not updates:
            raise ValueError("No updates provided for aggregation.")
        num_layers = len(updates[0][1])
        return [
            np.median(np.stack([params[i] for _, params in updates]), axis=0).astype(updates[0][1][i].dtype)
            for i in range(num_layers)
        ]


class TrimmedMeanStrategy(AggregationStrategy):
    """Coordinate-wise trimmed mean: drop the `trim_ratio` largest and smallest
    values per coordinate, average the rest (Yin et al., 2018)."""

    def __init__(self, trim_ratio: float = 0.1):
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5).")
        self.trim_ratio = trim_ratio

    def aggregate(self, updates: List[Tuple[int, NDArrays]]) -> NDArrays:
        if not updates:
            raise ValueError("No updates provided for aggregation.")
        n = len(updates)
        k = int(n * self.trim_ratio)
        num_layers = len(updates[0][1])
        out = []
        for i in range(num_layers):
            stacked = np.sort(np.stack([params[i] for _, params in updates]), axis=0)
            out.append(stacked[k:n - k].mean(axis=0).astype(updates[0][1][i].dtype))
        return out


# Krum / Multi-Krum are planned for Member 4's baseline work.
AGGREGATORS = {
    "fedavg": FedAvgStrategy,
    "median": MedianStrategy,
    "trimmed_mean": TrimmedMeanStrategy,
}


def build_aggregation_strategy(federation_config: dict) -> AggregationStrategy:
    """Build the aggregator named by config `federation.aggregation` (default fedavg)."""
    name = str(federation_config.get("aggregation", "fedavg")).lower()
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation '{name}'. Choose from {sorted(AGGREGATORS)}.")
    if name == "trimmed_mean":
        return TrimmedMeanStrategy(float(federation_config.get("trim_ratio", 0.1)))
    return AGGREGATORS[name]()

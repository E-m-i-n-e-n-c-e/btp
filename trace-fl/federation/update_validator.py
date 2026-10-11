from typing import List, Optional, Tuple
import numpy as np
from core.types import NDArrays


class UpdateValidator:
    """Validates client updates before aggregation.

    Structural checks (always on): example count > 0, non-empty, ndarray-only,
    float32 dtype, finite values, expected shapes.

    Optional hardening (off when None, so attacks stay observable by default):
      max_num_examples: reject updates claiming more samples than this. FedAvg
          weights by the *client-reported* num_examples, so without a cap one
          client can claim 10^9 samples and own the global model.
      max_update_norm: reject updates whose L2 distance from the current global
          model exceeds this (norm bounding; blunts scaling / large-noise attacks).
    """

    def __init__(
        self,
        expected_shapes: Optional[List[tuple]] = None,
        max_num_examples: Optional[int] = None,
        max_update_norm: Optional[float] = None,
    ):
        self.expected_shapes = expected_shapes
        self.max_num_examples = max_num_examples
        self.max_update_norm = max_update_norm

    def validate(self, client_id: str, num_examples: int, parameters: NDArrays,
                 global_parameters: Optional[NDArrays] = None) -> bool:
        """Returns True if the update is valid, False otherwise."""
        return self.check(num_examples, parameters, global_parameters)[0]

    def check(self, num_examples: int, parameters: NDArrays,
              global_parameters: Optional[NDArrays] = None) -> Tuple[bool, str]:
        """Returns (is_valid, reason); reason is "" for valid updates."""
        if num_examples <= 0:
            return False, "num_examples<=0"

        if self.max_num_examples is not None and num_examples > self.max_num_examples:
            return False, "num_examples_above_cap"

        if not parameters:
            return False, "empty_parameters"

        for p in parameters:
            if not isinstance(p, np.ndarray):
                return False, "non_ndarray"
            # Model weights are float32. An int array as the first update made
            # FedAvg's in-place float add raise and crash the whole round.
            if p.dtype != np.float32:
                return False, "dtype_not_float32"
            if not np.isfinite(p).all():
                return False, "non_finite"

        if self.expected_shapes is not None:
            if len(parameters) != len(self.expected_shapes):
                return False, "layer_count_mismatch"
            for p, expected_shape in zip(parameters, self.expected_shapes):
                if p.shape != expected_shape:
                    return False, "shape_mismatch"

        if self.max_update_norm is not None and global_parameters is not None:
            norm = update_l2_norm(parameters, global_parameters)
            if norm > self.max_update_norm:
                return False, "update_norm_above_bound"

        return True, ""


def update_l2_norm(parameters: NDArrays, global_parameters: NDArrays) -> float:
    """L2 norm of (w_client - w_global) across all layers."""
    return float(np.sqrt(sum(
        float(np.sum((p.astype(np.float64) - g) ** 2))
        for p, g in zip(parameters, global_parameters)
    )))

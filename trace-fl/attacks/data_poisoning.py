"""
Data poisoning attacks for TRACE-FL.

Includes LabelFlippingAttack implemented via a non-destructive PyTorch Dataset wrapper
that guarantees the underlying dataset in shared host memory is never mutated.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from attacks.base import AttackSchedule, DataPoisoningAttack


LABEL_FLIP_MODES = ("permutation", "targeted", "offset")


class LabelPoisonedDataset(Dataset):
    """
    Non-destructive decorator that wraps a PyTorch Dataset/Subset and applies
    label transformation on-the-fly during indexing.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        num_classes: int = 10,
        mode: str = "permutation",
        source_class: Optional[int] = None,
        target_class: Optional[int] = None,
        flip_fraction: float = 1.0,
        seed: int = 42,
    ):
        if mode not in LABEL_FLIP_MODES:
            raise ValueError(f"Unknown label-flip mode '{mode}'. Choose from {LABEL_FLIP_MODES}.")
        if mode == "targeted" and (source_class is None or target_class is None):
            # Previously this silently flipped nothing.
            raise ValueError("targeted label flipping needs params.source_class and params.target_class.")

        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.mode = mode
        self.source_class = source_class
        self.target_class = target_class
        self.flip_fraction = max(0.0, min(1.0, flip_fraction))
        self.seed = seed

        # Which indices get poisoned, decided once and deterministically from the seed
        # (was one torch.Generator per __getitem__ call).
        n = len(base_dataset)
        if self.flip_fraction >= 1.0:
            self._flip_mask = np.ones(n, dtype=bool)
        else:
            self._flip_mask = np.random.default_rng(seed).random(n) < self.flip_fraction

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _transform_label(self, y: int, index: int) -> int:
        if not self._flip_mask[index]:
            return y

        if self.mode == "permutation":
            # Swaps 0 <-> 9, 1 <-> 8, 2 <-> 7, etc.
            return (self.num_classes - 1) - y

        if self.mode == "targeted":
            return self.target_class if y == self.source_class else y

        # offset: shift label by 1 modulo num_classes
        return (y + 1) % self.num_classes

    def __getitem__(self, index: int):
        data, target = self.base_dataset[index]
        target_int = int(target)
        poisoned_target = self._transform_label(target_int, index)
        return data, poisoned_target


class LabelFlippingAttack(DataPoisoningAttack):
    """
    Label Flipping Attack.

    Poison the training dataset by permuting or retargeting labels during local training.
    Does not touch model weights post-training; the malicious gradient is learned naturally.
    """

    def __init__(
        self,
        schedule: AttackSchedule,
        intensity: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name="label_flipping",
            schedule=schedule,
            intensity=intensity,
            params=params,
        )
        self.num_classes = self.params.get("num_classes", 10)
        self.mode = self.params.get("mode", "permutation")
        self.source_class = self.params.get("source_class", None)
        self.target_class = self.params.get("target_class", None)
        # Validate at config-load time (the dataset wrapper re-checks per round).
        if self.mode not in LABEL_FLIP_MODES:
            raise ValueError(f"Unknown label-flip mode '{self.mode}'. Choose from {LABEL_FLIP_MODES}.")
        if self.mode == "targeted" and (self.source_class is None or self.target_class is None):
            raise ValueError("targeted label flipping needs params.source_class and params.target_class.")

    def poison_train_loader(
        self, train_loader: DataLoader, round_num: int
    ) -> DataLoader:
        """Wrap the DataLoader's dataset in LabelPoisonedDataset."""
        poisoned_dataset = LabelPoisonedDataset(
            base_dataset=train_loader.dataset,
            num_classes=self.num_classes,
            mode=self.mode,
            source_class=self.source_class,
            target_class=self.target_class,
            flip_fraction=self.intensity,
            seed=self.schedule.seed + round_num,
        )

        train_generator = torch.Generator().manual_seed(
            self.schedule.seed + round_num
        )

        return DataLoader(
            poisoned_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            generator=train_generator,
        )

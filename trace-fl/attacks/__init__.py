"""
TRACE-FL Attack Engine package.

Exposes:
- AttackEngine: Unified attack plugin for FL clients.
- AttackSchedule: Scheduling and client targeting logic.
- LabelFlippingAttack: Data-poisoning attack.
- GaussianNoiseAttack: Model-poisoning noise injection.
- SignFlipAttack: Model-poisoning gradient reversal.
- ScalingAttack: Model-poisoning boost/amplification.
"""

from attacks.base import (
    AttackSchedule,
    BaseAttack,
    DataPoisoningAttack,
    ModelPoisoningAttack,
)
from attacks.data_poisoning import (
    LabelFlippingAttack,
    LabelPoisonedDataset,
)
from attacks.engine import AttackEngine
from attacks.model_poisoning import (
    GaussianNoiseAttack,
    ScalingAttack,
    SignFlipAttack,
)

__all__ = [
    "AttackEngine",
    "AttackSchedule",
    "BaseAttack",
    "DataPoisoningAttack",
    "ModelPoisoningAttack",
    "LabelFlippingAttack",
    "LabelPoisonedDataset",
    "GaussianNoiseAttack",
    "SignFlipAttack",
    "ScalingAttack",
]

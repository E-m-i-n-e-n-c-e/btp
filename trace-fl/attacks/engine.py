"""
Central AttackEngine for TRACE-FL.

Acts as a clean, plug-in decorator attached to clients.
Dispatches pre-training (data poisoning) and post-training (model poisoning) attacks
based on user configuration without modifying core FL logic.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader

from attacks.base import (
    AttackSchedule,
    BaseAttack,
    DataPoisoningAttack,
    ModelPoisoningAttack,
)
from attacks.data_poisoning import LabelFlippingAttack
from attacks.model_poisoning import (
    GaussianNoiseAttack,
    ScalingAttack,
    SignFlipAttack,
)


class AttackEngine:
    """
    Plugin engine that attaches to any federated client and manages
    the execution of data poisoning and model poisoning attacks.
    """

    SUPPORTED_ATTACKS = {
        "label_flipping": LabelFlippingAttack,
        "gaussian_noise": GaussianNoiseAttack,
        "sign_flip": SignFlipAttack,
        "scaling": ScalingAttack,
    }

    def __init__(
        self,
        enabled: bool = False,
        attack: Optional[BaseAttack] = None,
    ):
        self.enabled = enabled
        self.attack = attack

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AttackEngine":
        """Instantiate an AttackEngine from the experiment configuration dictionary."""
        attack_config = config.get("attack", {})
        if not attack_config or not attack_config.get("enabled", False):
            return cls(enabled=False, attack=None)

        attack_type = attack_config.get("type", "sign_flip").lower()
        if attack_type not in cls.SUPPORTED_ATTACKS:
            raise ValueError(
                f"Unknown attack type '{attack_type}'. "
                f"Choose from {sorted(cls.SUPPORTED_ATTACKS.keys())}."
            )

        intensity = float(attack_config.get("intensity", 1.0))
        sched_cfg = attack_config.get("schedule", {})
        targets_cfg = attack_config.get("targets", {})
        params = attack_config.get("params", {})

        # Build schedule
        schedule = AttackSchedule(
            mode=sched_cfg.get("mode", "continuous"),
            start_round=int(sched_cfg.get("start_round", 1)),
            end_round=int(sched_cfg.get("end_round", 1000)),
            interval=int(sched_cfg.get("interval", 2)),
            rounds=[int(r) for r in sched_cfg.get("rounds", [])],
            probability=float(sched_cfg.get("probability", 0.5)),
            client_ids=targets_cfg.get("client_ids", None),
            fraction=(
                float(targets_cfg["fraction"])
                if "fraction" in targets_cfg and targets_cfg["fraction"] is not None
                else None
            ),
            seed=int(config.get("seed", 42)),
            target_mode=targets_cfg.get("mode"),
        )

        attack_cls = cls.SUPPORTED_ATTACKS[attack_type]
        attack_instance = attack_cls(
            schedule=schedule,
            intensity=intensity,
            params=params,
        )

        return cls(enabled=True, attack=attack_instance)

    def apply_pre_training(
        self,
        train_loader: DataLoader,
        round_num: int,
        client_id: Union[str, int],
        num_clients: int,
    ) -> Tuple[DataLoader, bool]:
        """
        Intercept and optionally poison the client training DataLoader before local training.

        Returns:
            Tuple of (effective_train_loader, is_attack_applied)
        """
        if (
            not self.enabled
            or self.attack is None
            or not isinstance(self.attack, DataPoisoningAttack)
        ):
            return train_loader, False

        if not self.attack.should_attack(round_num, client_id, num_clients):
            return train_loader, False

        poisoned_loader = self.attack.poison_train_loader(
            train_loader, round_num
        )
        return poisoned_loader, True

    def apply_post_training(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        round_num: int,
        client_id: Union[str, int],
        num_clients: int,
    ) -> Tuple[List[np.ndarray], bool, Dict[str, Any]]:
        """
        Intercept and optionally poison model parameters after local training.

        Returns:
            Tuple of (effective_parameters, is_attack_applied, telemetry_metrics)
        """
        if (
            not self.enabled
            or self.attack is None
            or not isinstance(self.attack, ModelPoisoningAttack)
        ):
            return local_parameters, False, {}

        if not self.attack.should_attack(round_num, client_id, num_clients):
            return local_parameters, False, {}

        poisoned_params, metrics = self.attack.poison_update(
            global_parameters=global_parameters,
            local_parameters=local_parameters,
            round_num=round_num,
            client_id=client_id,
        )
        metrics["attack_type"] = self.attack.name
        metrics["round"] = round_num
        metrics["client_id"] = str(client_id)

        return poisoned_params, True, metrics

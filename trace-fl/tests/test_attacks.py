"""
Unit and regression tests for Member 3 — Attack Engine.

Tests:
1. AttackSchedule (continuous, intermittent, client targeting)
2. LabelFlippingAttack & LabelPoisonedDataset (non-mutating data poisoning)
3. GaussianNoiseAttack (model poisoning)
4. SignFlipAttack (gradient reversal on delta)
5. ScalingAttack (gradient boosting on delta)
6. AttackEngine plugin lifecycle & hooks
7. Classification metrics (Macro-F1, Confusion Matrix, OvR ROC-AUC)
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from attacks.base import AttackSchedule
from attacks.data_poisoning import LabelFlippingAttack, LabelPoisonedDataset
from attacks.engine import AttackEngine
from attacks.model_poisoning import (
    GaussianNoiseAttack,
    ScalingAttack,
    SignFlipAttack,
)
from training.trainer import compute_classification_metrics


def test_attack_schedule_continuous():
    sched = AttackSchedule(
        mode="continuous",
        start_round=3,
        end_round=5,
        client_ids=[0],
    )
    assert not sched.is_active(round_num=2, client_id="client_0", num_clients=5)
    assert sched.is_active(round_num=3, client_id="client_0", num_clients=5)
    assert sched.is_active(round_num=4, client_id="client_0", num_clients=5)
    assert sched.is_active(round_num=5, client_id="client_0", num_clients=5)
    assert not sched.is_active(round_num=6, client_id="client_0", num_clients=5)
    # Honest client should never attack
    assert not sched.is_active(round_num=3, client_id="client_1", num_clients=5)


def test_attack_schedule_intermittent():
    sched = AttackSchedule(
        mode="intermittent",
        start_round=1,
        end_round=10,
        interval=2,
        client_ids=[0],
    )
    # R1: (1-1)%2 == 0 -> Active
    assert sched.is_active(round_num=1, client_id="client_0", num_clients=5)
    # R2: (2-1)%2 == 1 -> Inactive
    assert not sched.is_active(round_num=2, client_id="client_0", num_clients=5)
    # R3: (3-1)%2 == 0 -> Active
    assert sched.is_active(round_num=3, client_id="client_0", num_clients=5)
    # R4: (4-1)%2 == 1 -> Inactive
    assert not sched.is_active(round_num=4, client_id="client_0", num_clients=5)


def test_attack_schedule_fraction_targeting():
    sched = AttackSchedule(
        mode="continuous",
        start_round=1,
        end_round=10,
        fraction=0.4,  # 40% of 5 clients = 2 malicious clients (0, 1)
    )
    assert sched.is_active(round_num=1, client_id="client_0", num_clients=5)
    assert sched.is_active(round_num=1, client_id="client_1", num_clients=5)
    assert not sched.is_active(round_num=1, client_id="client_2", num_clients=5)
    assert not sched.is_active(round_num=1, client_id="client_3", num_clients=5)


def test_label_flipping_non_destructive():
    # Create synthetic dataset with labels 0 to 9
    x = torch.zeros((10, 1, 28, 28))
    y = torch.arange(0, 10, dtype=torch.int64)
    base_ds = TensorDataset(x, y)

    # Wrap in LabelPoisonedDataset (permutation: y -> 9 - y)
    poisoned_ds = LabelPoisonedDataset(base_ds, num_classes=10, mode="permutation")

    assert len(poisoned_ds) == 10

    # Verify flipped labels
    for i in range(10):
        _, poisoned_label = poisoned_ds[i]
        assert poisoned_label == (9 - i)

    # CRITICAL: Verify base dataset in memory is completely UNCHANGED
    for i in range(10):
        _, original_label = base_ds[i]
        assert original_label.item() == i


def test_label_flipping_targeted():
    x = torch.zeros((5, 1, 28, 28))
    y = torch.tensor([0, 7, 7, 2, 7])
    base_ds = TensorDataset(x, y)

    # Target 7 -> 1
    poisoned_ds = LabelPoisonedDataset(
        base_ds,
        num_classes=10,
        mode="targeted",
        source_class=7,
        target_class=1,
    )

    expected = [0, 1, 1, 2, 1]
    for i in range(5):
        _, p_y = poisoned_ds[i]
        assert p_y == expected[i]


def test_gaussian_noise_attack():
    sched = AttackSchedule(mode="continuous", start_round=1, end_round=10, client_ids=[0])
    attack = GaussianNoiseAttack(schedule=sched, intensity=1.0)

    # Global model weights
    global_p = [np.zeros((10, 5), dtype=np.float32)]
    # Local trained weights
    local_p = [np.ones((10, 5), dtype=np.float32)]

    poisoned_p, metrics = attack.poison_update(global_p, local_p, round_num=1)

    assert len(poisoned_p) == 1
    assert poisoned_p[0].shape == (10, 5)
    # Weights should not be identical to local_p
    assert not np.allclose(poisoned_p[0], local_p[0])
    assert "l2_norm_honest" in metrics
    assert "l2_norm_poisoned" in metrics
    assert metrics["l2_norm_honest"] == pytest.approx(float(np.sqrt(50)), rel=1e-3)


def test_sign_flip_attack():
    sched = AttackSchedule(mode="continuous", start_round=1, end_round=10, client_ids=[0])
    attack = SignFlipAttack(schedule=sched, intensity=1.0)

    # w_global = 2.0, w_local = 3.0 -> delta = +1.0
    global_p = [np.full((4, 4), 2.0, dtype=np.float32)]
    local_p = [np.full((4, 4), 3.0, dtype=np.float32)]

    poisoned_p, metrics = attack.poison_update(global_p, local_p, round_num=1)

    # Inverted delta should be -1.0 -> poisoned = w_global + (-1.0) = 1.0
    expected = np.full((4, 4), 1.0, dtype=np.float32)
    assert np.allclose(poisoned_p[0], expected)
    # Cosine similarity must be exactly -1.0
    assert metrics["cosine_similarity"] == pytest.approx(-1.0, abs=1e-5)


def test_scaling_attack():
    sched = AttackSchedule(mode="continuous", start_round=1, end_round=10, client_ids=[0])
    attack = ScalingAttack(schedule=sched, intensity=10.0)

    # w_global = 1.0, w_local = 1.5 -> delta = +0.5
    global_p = [np.full((2, 2), 1.0, dtype=np.float32)]
    local_p = [np.full((2, 2), 1.5, dtype=np.float32)]

    poisoned_p, metrics = attack.poison_update(global_p, local_p, round_num=1)

    # Scaled delta = 0.5 * 10 = 5.0 -> poisoned = 1.0 + 5.0 = 6.0
    expected = np.full((2, 2), 6.0, dtype=np.float32)
    assert np.allclose(poisoned_p[0], expected)
    # Cosine similarity is positive 1.0 (same direction, larger norm)
    assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["l2_norm_poisoned"] == pytest.approx(metrics["l2_norm_honest"] * 10.0, rel=1e-3)


def test_attack_engine_plugin():
    config = {
        "num_clients": 5,
        "seed": 42,
        "attack": {
            "enabled": True,
            "type": "sign_flip",
            "intensity": 1.0,
            "schedule": {
                "mode": "continuous",
                "start_round": 1,
                "end_round": 5,
            },
            "targets": {
                "mode": "client_ids",
                "client_ids": [0],
            },
        },
    }

    engine = AttackEngine.from_config(config)
    assert engine.enabled

    # Attacking client 0
    w_glob = [np.ones((2, 2), dtype=np.float32)]
    w_loc = [np.full((2, 2), 2.0, dtype=np.float32)]

    out_p, attacked, _ = engine.apply_post_training(
        global_parameters=w_glob,
        local_parameters=w_loc,
        round_num=1,
        client_id="client_0",
        num_clients=5,
    )
    assert attacked
    assert np.allclose(out_p[0], np.zeros((2, 2)))

    # Honest client 1
    out_p_honest, attacked_honest, _ = engine.apply_post_training(
        global_parameters=w_glob,
        local_parameters=w_loc,
        round_num=1,
        client_id="client_1",
        num_clients=5,
    )
    assert not attacked_honest
    assert np.allclose(out_p_honest[0], w_loc[0])


def test_classification_metrics_computation():
    # Perfect predictor test
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    probs = torch.eye(10)

    metrics = compute_classification_metrics(labels, preds, probs, num_classes=10)

    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["roc_auc"]["macro_auc"] == 1.0
    for c in range(10):
        assert metrics["class_f1"][c] == 1.0
        assert metrics["roc_auc"]["class_auc"][c] == 1.0

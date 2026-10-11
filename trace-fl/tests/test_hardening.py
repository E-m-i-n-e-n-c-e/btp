"""Regression tests for the fixes in the attack/validation review."""

import importlib
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from attacks.base import AttackSchedule, parse_client_index
from attacks.data_poisoning import LabelFlippingAttack, LabelPoisonedDataset
from attacks.engine import AttackEngine
from attacks.model_poisoning import GaussianNoiseAttack, ScalingAttack
from federation.strategy import (
    MedianStrategy,
    TrimmedMeanStrategy,
    build_aggregation_strategy,
)
from federation.update_stats import compute_update_stats
from federation.update_validator import UpdateValidator


@pytest.mark.parametrize("module", ["app.server_app", "app.client_app", "telemetry.logging"])
def test_flower_entrypoints_import(module):
    # Missing `Optional` import in telemetry.logging used to crash every run here.
    importlib.import_module(module)


def test_parse_client_index():
    assert parse_client_index("client_3") == 3
    assert parse_client_index("3") == 3
    assert parse_client_index(3) == 3
    assert parse_client_index("abc") is None


def test_unparseable_client_is_never_targeted():
    # Used to map to index 0, i.e. the default attacker.
    sched = AttackSchedule(client_ids=[0])
    assert not sched.is_active(1, "8f3a-opaque-ray-id", 5)


@pytest.mark.parametrize("kwargs", [
    {"interval": 0},
    {"mode": "sometimes"},
    {"start_round": 5, "end_round": 2},
    {"probability": 1.5},
    {"fraction": 0.0},
    {"target_mode": "fraction"},  # fraction missing
])
def test_invalid_schedule_rejected(kwargs):
    with pytest.raises(ValueError):
        AttackSchedule(**kwargs)


def test_probabilistic_mode_accepts_any_id_format():
    sched = AttackSchedule(mode="probabilistic", probability=1.0, client_ids=[2])
    assert sched.is_active(1, "2", 5)  # used to raise on ids without '_'


def test_targets_mode_fraction_overrides_leftover_client_ids():
    # run.py --attack-clients 0.4 sets mode=fraction but config still has client_ids: [0].
    cfg = {"num_clients": 5, "attack": {
        "enabled": True, "type": "sign_flip",
        "targets": {"mode": "fraction", "fraction": 0.4, "client_ids": [0]},
    }}
    sched = AttackEngine.from_config(cfg).attack.schedule
    assert sched.malicious_indices(5) == [0, 1]


def test_gaussian_noise_differs_between_attackers():
    sched = AttackSchedule(client_ids=[0, 1])
    attack = GaussianNoiseAttack(schedule=sched, intensity=1.0)
    g = [np.zeros((50,), dtype=np.float32)]
    loc = [np.ones((50,), dtype=np.float32)]
    p0, _ = attack.poison_update(g, loc, round_num=1, client_id="client_0")
    p1, _ = attack.poison_update(g, loc, round_num=1, client_id="client_1")
    assert not np.allclose(p0[0], p1[0])


def test_scaling_noop_warns():
    with pytest.warns(UserWarning, match="no-op"):
        ScalingAttack(schedule=AttackSchedule(), intensity=1.0)


def test_label_flip_invalid_configs_rejected():
    with pytest.raises(ValueError):
        LabelFlippingAttack(AttackSchedule(), params={"mode": "targeted"})
    with pytest.raises(ValueError):
        LabelFlippingAttack(AttackSchedule(), params={"mode": "rotate"})


def test_label_flip_fraction_is_deterministic_and_partial():
    ds = TensorDataset(torch.zeros(2000, 1), torch.zeros(2000, dtype=torch.int64))
    a = LabelPoisonedDataset(ds, flip_fraction=0.3, seed=7)
    b = LabelPoisonedDataset(ds, flip_fraction=0.3, seed=7)
    labels_a = [a[i][1] for i in range(2000)]
    assert labels_a == [b[i][1] for i in range(2000)]
    assert 0.25 < sum(1 for y in labels_a if y == 9) / 2000 < 0.35


def test_validator_caps():
    g = [np.zeros((4,), dtype=np.float32)]
    v = UpdateValidator(expected_shapes=[(4,)], max_num_examples=100, max_update_norm=1.0)
    assert v.check(50, [np.full((4,), 0.1, dtype=np.float32)], g) == (True, "")
    assert v.check(10**9, [np.zeros((4,), dtype=np.float32)], g)[1] == "num_examples_above_cap"
    assert v.check(50, [np.full((4,), 10.0, dtype=np.float32)], g)[1] == "update_norm_above_bound"
    assert v.check(50, [np.array([np.nan] * 4, dtype=np.float32)], g)[1] == "non_finite"
    assert v.check(50, [np.zeros((4,), dtype=np.int64)], g)[1] == "dtype_not_float32"


def test_median_and_trimmed_mean_resist_outlier():
    honest = [(10, [np.full((3,), 1.0, dtype=np.float32)]) for _ in range(4)]
    attacker = [(10, [np.full((3,), 1000.0, dtype=np.float32)])]
    assert np.allclose(MedianStrategy().aggregate(honest + attacker)[0], 1.0)
    assert np.allclose(TrimmedMeanStrategy(0.2).aggregate(honest + attacker)[0], 1.0)
    with pytest.raises(ValueError):
        build_aggregation_strategy({"aggregation": "krum"})


def test_update_stats_flags_sign_flip():
    g = [np.zeros((3,))]
    honest = [[np.array([1.0, 1.0, 0.1 * i])] for i in range(5)]
    flipped = [[np.array([-1.0, -1.0, 0.0])]]
    stats = compute_update_stats(g, honest + flipped)
    assert all(s["cosine_to_peers"] > 0.5 for s in stats[:5])
    assert stats[5]["cosine_to_peers"] < -0.9

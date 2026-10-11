import pytest
from data.dataset import FederatedDatasetProvider


def test_dataset_partitioning():
    provider = FederatedDatasetProvider(
        num_clients=2,
        batch_size=32,
        seed=42,
        dataset_name="mnist",
        partition_type="iid",
    )

    partition_0 = provider.get_partition(0)
    partition_1 = provider.get_partition(1)

    assert partition_0.partition_id == 0
    assert partition_1.partition_id == 1
    assert partition_0.num_examples > 0
    assert partition_1.num_examples > 0
    assert partition_0.num_examples + partition_1.num_examples == 60000


def test_fashion_mnist_partitioning():
    provider = FederatedDatasetProvider(
        num_clients=2,
        batch_size=32,
        seed=42,
        dataset_name="fashion_mnist",
        partition_type="iid",
    )

    partition_0 = provider.get_partition(0)
    partition_1 = provider.get_partition(1)

    assert partition_0.partition_id == 0
    assert partition_1.partition_id == 1
    assert partition_0.num_examples + partition_1.num_examples == 60000


def test_dirichlet_partitioning_alphas():
    # Test Dirichlet with alpha values from DatasetSplit_Documentation.pdf (1.0, 0.5, 0.1)
    for alpha in (1.0, 0.5, 0.1):
        provider = FederatedDatasetProvider(
            num_clients=5,
            batch_size=32,
            seed=42,
            dataset_name="mnist",
            partition_type="dirichlet",
            alpha=alpha,
        )
        total_examples = sum(
            provider.get_partition(i).num_examples for i in range(5)
        )
        assert total_examples == 60000


def test_invalid_partition():
    provider = FederatedDatasetProvider(num_clients=2, batch_size=32, seed=42)
    with pytest.raises(ValueError):
        provider.get_partition(2)


def test_invalid_dataset_name():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        FederatedDatasetProvider(
            num_clients=2,
            batch_size=32,
            dataset_name="cifar10",
        )


def test_invalid_partition_type():
    with pytest.raises(ValueError, match="Unsupported partition_type"):
        FederatedDatasetProvider(
            num_clients=2,
            batch_size=32,
            partition_type="pathological",
        )


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be greater than 0"):
        FederatedDatasetProvider(
            num_clients=2,
            batch_size=32,
            partition_type="dirichlet",
            alpha=0.0,
        )

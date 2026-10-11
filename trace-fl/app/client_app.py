import torch
import time

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from core.config import load_config
from core.model import create_model, set_parameters, get_parameters
from data.dataset import FederatedDatasetProvider
from training.trainer import train, evaluate
from telemetry.communication import CommunicationTracker
from telemetry.logging import log_client, log_client_evaluate, log_attack
from attacks.engine import AttackEngine


class TRACEClient(NumPyClient):
    def __init__(self, client_id: str, partition_id: int, config: dict):
        self.client_id = client_id
        self.partition_id = partition_id
        self.config = config

        self.model = create_model()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # ------------------------------------------------------------------
        # Dataset configuration
        # ------------------------------------------------------------------
        dataset_config = config.get("dataset", {})

        dataset_name = dataset_config.get("name", "mnist")
        partition_type = dataset_config.get("partition_type", "iid")
        alpha = dataset_config.get("alpha", 0.5)

        # ------------------------------------------------------------------
        # Load client data partition
        # ------------------------------------------------------------------
        self.dataset_provider = FederatedDatasetProvider(
            num_clients=config.get("num_clients", 20),
            batch_size=config.get("local_training", {}).get("batch_size", 32),
            seed=config.get("seed", 42),
            dataset_name=dataset_name,
            partition_type=partition_type,
            alpha=alpha,
        )

        self.partition = self.dataset_provider.get_partition(
            self.partition_id
        )

        self.tracker = CommunicationTracker()
        self.attack_engine = AttackEngine.from_config(self.config)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        start_time = time.time()
        round_num = int(config.get("round", 1))

        # Track downlink
        downlink_bytes = self.tracker.record_downlink(
            round_num,
            self.client_id,
            parameters,
        )

        # Update model with global parameters
        set_parameters(self.model, parameters)

        # Apply pre-training attack (e.g. Label Flipping data poisoning)
        train_loader, attack_applied_pre = self.attack_engine.apply_pre_training(
            train_loader=self.partition.train_loader,
            round_num=round_num,
            client_id=self.client_id,
            num_clients=self.config.get("num_clients", 20),
        )

        # Local training configuration
        epochs = self.config.get("local_training", {}).get("epochs", 1)
        learning_rate = self.config.get(
            "local_training", {}
        ).get("learning_rate", 0.01)

        # Train locally
        metrics = train(
            model=self.model,
            train_loader=train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device,
        )

        # Get updated parameters
        raw_local_parameters = get_parameters(self.model)

        # Apply post-training attack (e.g. Gaussian Noise, Sign Flip, Scaling)
        final_parameters, attack_applied_post, attack_telemetry = self.attack_engine.apply_post_training(
            global_parameters=parameters,
            local_parameters=raw_local_parameters,
            round_num=round_num,
            client_id=self.client_id,
            num_clients=self.config.get("num_clients", 20),
        )

        # Track uplink with final transmitted parameters
        uplink_bytes = self.tracker.record_uplink(
            round_num,
            self.client_id,
            final_parameters,
        )

        training_time = time.time() - start_time

        # Log client execution
        log_client(
            client_id=self.client_id,
            partition_id=self.partition_id,
            num_examples=self.partition.num_examples,
            local_epochs=epochs,
            training_time=training_time,
            local_loss=metrics["loss"],
            local_accuracy=metrics["accuracy"],
            uplink_bytes=uplink_bytes,
            downlink_bytes=downlink_bytes,
        )

        # If attack was applied, log attack telemetry event
        if attack_applied_pre:
            log_attack(
                client_id=self.client_id,
                round_num=round_num,
                attack_type=self.attack_engine.attack.name if self.attack_engine.attack else "data_poisoning",
                metrics={"mode": getattr(self.attack_engine.attack, "mode", "label_flipping")},
            )
        elif attack_applied_post:
            log_attack(
                client_id=self.client_id,
                round_num=round_num,
                attack_type=self.attack_engine.attack.name if self.attack_engine.attack else "model_poisoning",
                metrics=attack_telemetry,
            )

        return (
            final_parameters,
            self.partition.num_examples,
            {
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "client_id": self.client_id,
                "is_attacked": attack_applied_pre or attack_applied_post,
            },
        )

    def evaluate(self, parameters, config):
        # Update model
        set_parameters(self.model, parameters)

        # Evaluate on client's local test partition
        loss, accuracy = evaluate(
            model=self.model,
            test_loader=self.partition.test_loader,
            device=self.device,
        )

        # Log evaluation
        log_client_evaluate(
            client_id=self.client_id,
            partition_id=self.partition_id,
            num_examples=self.partition.num_examples,
            loss=float(loss),
            accuracy=float(accuracy),
        )

        return (
            float(loss),
            self.partition.num_examples,
            {"accuracy": float(accuracy), "client_id": self.client_id},
        )


def client_fn(context: Context):
    """Flower ClientApp entry point."""

    # Resolved relative to the code, not the CWD (which differs between
    # simulation and deployment containers).
    config = load_config()

    # Flower simulation provides partition-id (0..N-1); deployment passes it via
    # --node-config. context.node_id is a random 64-bit int, so `node_id % N`
    # would silently give two clients the same partition -> fail loudly instead.
    if "partition-id" not in context.node_config:
        raise RuntimeError(
            "node_config has no 'partition-id'. In deployment mode start the "
            "SuperNode with --node-config \"partition-id=<i> num-partitions=<N>\"."
        )
    partition_id = int(context.node_config["partition-id"])

    client_id = f"client_{partition_id}"

    return TRACEClient(
        client_id=client_id,
        partition_id=partition_id,
        config=config,
    ).to_client()


app = ClientApp(client_fn=client_fn)
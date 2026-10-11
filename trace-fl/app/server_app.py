from core.config import load_config
import time
import torch

from typing import Any, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context

from core.model import create_model, get_parameters, set_parameters
from federation.strategy import build_aggregation_strategy
from federation.client_selector import RandomClientSelector
from federation.update_validator import UpdateValidator
from federation.update_stats import compute_update_stats
from federation.round_state import RoundState
from telemetry.communication import CommunicationTracker
from telemetry.logging import log_client_update, log_round, logger
from training.trainer import evaluate as evaluate_model
from data.dataset import FederatedDatasetProvider


class TRACEStrategy(Strategy):
    def __init__(self, config: dict):
        self.config = config
        federation_cfg = self.config.get("federation", {})
        self.aggregation_strategy = build_aggregation_strategy(federation_cfg)

        # In simulation, clients are sequential indices.
        seed = self.config.get("seed", 42)
        self.client_selector = RandomClientSelector(seed=seed)

        # Update validator expects parameter shapes matching the global model.
        self.global_model = create_model()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        initial_params = get_parameters(self.global_model)
        expected_shapes = [p.shape for p in initial_params]

        self.update_validator = UpdateValidator(
            expected_shapes=expected_shapes,
            max_num_examples=federation_cfg.get("max_num_examples"),
            max_update_norm=federation_cfg.get("max_update_norm"),
        )
        # Global weights sent out this round; reference point for update deltas.
        self.current_global: Optional[List] = None

        self.round_state = RoundState()
        self.tracker = CommunicationTracker()

        self.min_available_clients = self.config.get(
            "federation", {}
        ).get("min_available_clients", 20)

        self.fraction_train = self.config.get(
            "federation", {}
        ).get("fraction_train", 1.0)

        # --------------------------------------------------------------
        # Dataset configuration
        # --------------------------------------------------------------
        dataset_config = self.config.get("dataset", {})

        dataset_name = dataset_config.get("name", "mnist")
        partition_type = dataset_config.get("partition_type", "iid")
        alpha = dataset_config.get("alpha", 0.5)

        # --------------------------------------------------------------
        # Global test dataset
        #
        # Must use the same dataset configuration as the clients.
        # --------------------------------------------------------------
        self.dataset_provider = FederatedDatasetProvider(
            num_clients=self.config.get("num_clients", 20),
            batch_size=self.config.get(
                "local_training", {}
            ).get("batch_size", 32),
            seed=seed,
            dataset_name=dataset_name,
            partition_type=partition_type,
            alpha=alpha,
        )

        self.test_loader = self.dataset_provider.get_global_test_loader()
        self.eval_history: List[Dict[str, Any]] = []

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        ndarrays = get_parameters(self.global_model)

        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        self.round_state.start_round(server_round, [])
        self.round_state.round_start_time = time.time()

        # Sample clients.
        sample_size = int(
            client_manager.num_available() * self.fraction_train
        )
        sample_size = max(sample_size, 1)

        available_clients_dict = client_manager.all()

        # Ensure we have enough clients.
        if len(available_clients_dict) < self.min_available_clients:
            return []

        # Use custom client selector.
        cids = list(available_clients_dict.keys())

        selected_cids = self.client_selector.select(
            cids,
            sample_size,
        )

        clients = [
            available_clients_dict[cid]
            for cid in selected_cids
        ]

        self.round_state.selected_clients = selected_cids

        # Track downlink communication.
        ndarrays = parameters_to_ndarrays(parameters)
        self.current_global = ndarrays

        for cid in selected_cids:
            self.tracker.record_downlink(
                server_round,
                cid,
                ndarrays,
            )

        fit_ins = FitIns(
            parameters,
            {"round": server_round},
        )

        return [
            (client, fit_ins)
            for client in clients
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[
            Union[
                Tuple[ClientProxy, FitRes],
                BaseException,
            ]
        ],
    ) -> Tuple[
        Optional[Parameters],
        Dict[str, Scalar],
    ]:
        """Aggregate training results."""

        agg_start_time = time.time()

        valid_updates = []
        received = []  # (cid_str, fit_res, ndarrays, is_valid, reason)

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(
                fit_res.parameters
            )

            # Log client's local metrics.
            metrics = fit_res.metrics

            # Self-reported, human-readable ID (Ray's client.cid is an opaque int).
            cid_str = str(metrics.get(
                "client_id",
                client.cid,
            ))

            logger.info(
                f"Client {cid_str} local training complete -> "
                f"Loss: {float(metrics.get('loss', 0.0)):.4f} | "
                f"Accuracy: {float(metrics.get('accuracy', 0.0)):.4f}"
            )

            # Track uplink.
            self.tracker.record_uplink(
                server_round,
                cid_str,
                ndarrays,
            )

            # Validate update.
            is_valid, reason = self.update_validator.check(
                num_examples=fit_res.num_examples,
                parameters=ndarrays,
                global_parameters=self.current_global,
            )
            received.append((cid_str, fit_res, ndarrays, is_valid, reason))

            if is_valid:
                valid_updates.append(
                    (
                        fit_res.num_examples,
                        ndarrays,
                    )
                )

                self.round_state.successful_clients.append(cid_str)

            else:
                logger.warning(f"Rejected update from {cid_str}: {reason}")
                self.round_state.failed_clients.append(cid_str)

        self._log_update_signals(server_round, received)

        if not valid_updates:
            return None, {}

        # Aggregate using FedAvg.
        aggregated_ndarrays = self.aggregation_strategy.aggregate(
            valid_updates
        )

        self.round_state.aggregation_time = (
            time.time() - agg_start_time
        )

        return (
            ndarrays_to_parameters(aggregated_ndarrays),
            {},
        )

    def _log_update_signals(self, server_round: int, received: list) -> None:
        """Emit one client_update event per received update (norm + cosine)."""
        if self.current_global is None:
            return
        # Only finite, correctly shaped updates can be compared numerically.
        comparable = [r for r in received if r[3] or r[4] in (
            "num_examples_above_cap", "update_norm_above_bound")]
        stats = compute_update_stats(self.current_global, [r[2] for r in comparable])
        for (cid_str, fit_res, _, is_valid, reason), st in zip(comparable, stats):
            log_client_update(
                round_num=server_round,
                client_id=cid_str,
                num_examples=fit_res.num_examples,
                accepted=is_valid,
                rejection_reason=reason,
                is_attacked=bool(fit_res.metrics.get("is_attacked", False)),
                update_norm=st["update_norm"],
                cosine_to_peers=st["cosine_to_peers"],
            )

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Client-side evaluation is disabled.
        Global server-side evaluation is used instead.
        """

        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[
            Union[
                Tuple[ClientProxy, EvaluateRes],
                BaseException,
            ]
        ],
    ) -> Tuple[
        Optional[float],
        Dict[str, Scalar],
    ]:
        """Aggregate evaluation results."""

        if not results:
            return None, {}

        total_examples = sum(
            res.num_examples
            for _, res in results
        )

        # Weighted average of loss.
        agg_loss = sum(
            res.loss * res.num_examples
            for _, res in results
        ) / total_examples

        # Weighted average of accuracy.
        agg_acc = sum(
            float(
                res.metrics.get("accuracy", 0.0)
            ) * res.num_examples
            for _, res in results
        ) / total_examples

        return agg_loss, {
            "accuracy": agg_acc
        }

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[
        Tuple[
            float,
            Dict[str, Scalar],
        ]
    ]:
        """Evaluate global model on the global test dataset."""

        ndarrays = parameters_to_ndarrays(
            parameters
        )

        set_parameters(
            self.global_model,
            ndarrays,
        )

        loss, accuracy, detailed = evaluate_model(
            self.global_model,
            self.test_loader,
            self.device,
            return_details=True,
        )

        macro_f1 = float(detailed.get("macro_f1", 0.0))
        roc_auc_macro = float(detailed.get("roc_auc", {}).get("macro_auc", 0.0))

        self.round_state.global_loss = loss
        self.round_state.global_accuracy = accuracy

        # Record evaluation stats for this round
        eval_record = {
            "round": server_round,
            "global_loss": loss,
            "global_accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": float(detailed.get("weighted_f1", 0.0)),
            "class_f1": detailed.get("class_f1", {}),
            "class_precision": detailed.get("class_precision", {}),
            "class_recall": detailed.get("class_recall", {}),
            "roc_auc": detailed.get("roc_auc", {}),
            "confusion_matrix": detailed.get("confusion_matrix", []),
            "roc_curves": detailed.get("roc_curves", {}),
        }
        self.eval_history.append(eval_record)

        # Write latest evaluation history to JSON
        try:
            import json
            from pathlib import Path
            eval_dir = Path("artifacts/eval_results")
            eval_dir.mkdir(parents=True, exist_ok=True)
            with open(eval_dir / "eval_metrics.json", "w") as f:
                json.dump(self.eval_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist eval_metrics.json: {e}")

        if server_round == 0:
            self.round_state.round_duration = 0.0
        else:
            self.round_state.round_duration = (
                time.time() - self.round_state.round_start_time
            )

        round_totals = self.tracker.get_round_totals(
            server_round
        )

        # Log round telemetry with F1 and ROC AUC
        log_round(
            round_num=server_round,
            selected_clients=self.round_state.selected_clients,
            successful_clients=self.round_state.successful_clients,
            failed_clients=self.round_state.failed_clients,
            global_loss=loss,
            global_accuracy=accuracy,
            uplink_bytes=round_totals["uplink_bytes"],
            downlink_bytes=round_totals["downlink_bytes"],
            aggregation_time=self.round_state.aggregation_time,
            round_time=self.round_state.round_duration,
            macro_f1=macro_f1,
            roc_auc=roc_auc_macro,
        )

        return loss, {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "roc_auc": roc_auc_macro,
        }



def server_fn(context: Context) -> ServerAppComponents:
    """Flower ServerApp entry point."""
    config = load_config()
    num_rounds = config.get("num_rounds", 20)

    strategy = TRACEStrategy(config)
    config_obj = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        strategy=strategy,
        config=config_obj,
    )


app = ServerApp(server_fn=server_fn)
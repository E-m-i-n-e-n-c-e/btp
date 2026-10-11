import logging
import json
from typing import Any, Dict, List, Optional

def setup_logger(name: str = "TRACE-FL") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

logger = setup_logger()

def log_round(
    round_num: int, 
    selected_clients: List[str], 
    successful_clients: List[str],
    failed_clients: List[str],
    global_loss: float,
    global_accuracy: float,
    uplink_bytes: int,
    downlink_bytes: int,
    aggregation_time: float,
    round_time: float,
    macro_f1: Optional[float] = None,
    roc_auc: Optional[float] = None,
):
    """Log structured round telemetry."""
    log_data = {
        "event": "round_complete",
        "round": round_num,
        "selected_clients": len(selected_clients),
        "successful_clients": len(successful_clients),
        "failed_clients": len(failed_clients),
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes,
        "total_payload_bytes": uplink_bytes + downlink_bytes,
        "aggregation_time": aggregation_time,
        "round_time": round_time
    }
    if macro_f1 is not None:
        log_data["macro_f1"] = macro_f1
    if roc_auc is not None:
        log_data["roc_auc"] = roc_auc
    logger.info(json.dumps(log_data))


def log_client(
    client_id: str,
    partition_id: int,
    num_examples: int,
    local_epochs: int,
    training_time: float,
    local_loss: float,
    local_accuracy: float,
    uplink_bytes: int,
    downlink_bytes: int
):
    """Log structured client telemetry."""
    log_data = {
        "event": "client_complete",
        "client_id": client_id,
        "partition_id": partition_id,
        "num_examples": num_examples,
        "local_epochs": local_epochs,
        "training_time": training_time,
        "local_loss": local_loss,
        "local_accuracy": local_accuracy,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes
    }
    logger.info(json.dumps(log_data))

def log_client_evaluate(
    client_id: str,
    partition_id: int,
    num_examples: int,
    loss: float,
    accuracy: float
):
    """Log structured client evaluation telemetry."""
    log_data = {
        "event": "client_evaluate",
        "client_id": client_id,
        "partition_id": partition_id,
        "num_examples": num_examples,
        "global_model_loss_on_local_data": loss,
        "global_model_accuracy_on_local_data": accuracy
    }
    logger.info(json.dumps(log_data))


def log_attack(
    client_id: str,
    round_num: int,
    attack_type: str,
    metrics: Dict[str, Any],
):
    """Log structured attack event telemetry."""
    log_data = {
        "event": "attack_injected",
        "client_id": client_id,
        "round": round_num,
        "attack_type": attack_type,
        **metrics,
    }
    logger.info(json.dumps(log_data))



def log_client_update(
    round_num: int,
    client_id: str,
    num_examples: int,
    accepted: bool,
    rejection_reason: str,
    is_attacked: bool,
    update_norm: float,
    cosine_to_peers: float,
):
    """Server-side view of one client update (norm / cosine signals).

    is_attacked is the client's self-reported ground truth: it exists only so
    analysis can label honest vs malicious. A defense must never read it.
    """
    log_data = {
        "event": "client_update",
        "round": round_num,
        "client_id": client_id,
        "num_examples": num_examples,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "is_attacked": is_attacked,
        "update_norm": update_norm,
        "cosine_to_peers": cosine_to_peers,
    }
    logger.info(json.dumps(log_data))

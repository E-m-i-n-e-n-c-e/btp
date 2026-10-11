import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, Union, List, Optional

from core.model import MNISTNet


def train(
    model: MNISTNet,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train the model locally.

    Args:
        model: Model to train.
        train_loader: Client training DataLoader.
        epochs: Number of local training epochs.
        learning_rate: SGD learning rate.
        device: Device used for training.

    Returns:
        Dictionary containing average training loss and accuracy.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear previous gradients.
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(inputs)

            # Calculate loss.
            loss = criterion(outputs, labels)

            # Backward pass.
            loss.backward()

            # Update model parameters.
            optimizer.step()

            # Accumulate metrics.
            total_loss += loss.item() * inputs.size(0)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
        }

    average_loss = total_loss / total
    accuracy = correct / total

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


def compute_classification_metrics(
    all_labels: torch.Tensor,
    all_preds: torch.Tensor,
    all_probs: torch.Tensor,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Compute Macro/Weighted F1, Confusion Matrix, and OvR ROC curves."""
    import numpy as np

    # Vectorised confusion matrix (rows = true class, cols = predicted).
    cm = torch.bincount(
        all_labels.long() * num_classes + all_preds.long(),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    class_f1 = {}
    class_precision = {}
    class_recall = {}
    supports = []
    f1_list = []

    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        support = cm[c, :].sum().item()
        supports.append(support)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        class_precision[c] = round(prec, 4)
        class_recall[c] = round(rec, 4)
        class_f1[c] = round(f1, 4)
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    total_support = max(1, sum(supports))
    weighted_f1 = float(sum(f * s for f, s in zip(f1_list, supports)) / total_support)

    # One-vs-Rest ROC curve and AUC per class
    roc_curves = {}
    class_auc = {}
    labels_np = all_labels.cpu().numpy()
    probs_np = all_probs.cpu().numpy()

    for c in range(num_classes):
        y_true = (labels_np == c).astype(int)
        y_score = probs_np[:, c]

        P = max(1, int(np.sum(y_true == 1)))
        N = max(1, int(np.sum(y_true == 0)))

        # 51 threshold points from 1.0 down to 0.0
        thresholds = np.linspace(1.0, 0.0, 51)
        fprs = [0.0]
        tprs = [0.0]

        for t in thresholds:
            pred_pos = y_score >= t
            tp = int(np.sum(pred_pos & (y_true == 1)))
            fp = int(np.sum(pred_pos & (y_true == 0)))
            tprs.append(float(tp / P))
            fprs.append(float(fp / N))

        tprs.append(1.0)
        fprs.append(1.0)

        # Sort by FPR for trapezoidal integration
        sorted_idx = np.argsort(fprs)
        fprs_sorted = np.array(fprs)[sorted_idx]
        tprs_sorted = np.array(tprs)[sorted_idx]
        auc_val = float(np.trapz(tprs_sorted, fprs_sorted))

        class_auc[c] = round(auc_val, 4)
        roc_curves[c] = {
            "fpr": [round(float(x), 4) for x in fprs_sorted],
            "tpr": [round(float(x), 4) for x in tprs_sorted],
        }

    macro_auc = float(np.mean(list(class_auc.values())))

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "class_f1": class_f1,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "confusion_matrix": cm.tolist(),
        "roc_auc": {
            "macro_auc": round(macro_auc, 4),
            "class_auc": class_auc,
        },
        "roc_curves": roc_curves,
    }


def evaluate(
    model: MNISTNet,
    test_loader: DataLoader,
    device: torch.device,
    return_details: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
    """
    Evaluate the model on a test set.

    Args:
        model: Model to evaluate.
        test_loader: Evaluation DataLoader.
        device: Device used for evaluation.
        return_details: If True, also computes and returns detailed metrics
                        (Macro-F1, Weighted-F1, Confusion Matrix, OvR ROC curves).

    Returns:
        If return_details is False:
            Tuple of (average_loss, accuracy)
        If return_details is True:
            Tuple of (average_loss, accuracy, detailed_metrics_dict)
    """

    criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(inputs)

            # Calculate loss.
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_details:
                all_labels.append(labels.cpu())
                all_preds.append(predicted.cpu())
                all_probs.append(probs.cpu())

    if total == 0:
        if return_details:
            return 0.0, 0.0, {}
        return 0.0, 0.0

    average_loss = total_loss / total
    accuracy = correct / total

    if not return_details:
        return average_loss, accuracy

    labels_tensor = torch.cat(all_labels, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    probs_tensor = torch.cat(all_probs, dim=0)

    detailed_metrics = compute_classification_metrics(
        all_labels=labels_tensor,
        all_preds=preds_tensor,
        all_probs=probs_tensor,
    )
    detailed_metrics["loss"] = average_loss
    detailed_metrics["accuracy"] = accuracy

    return average_loss, accuracy, detailed_metrics
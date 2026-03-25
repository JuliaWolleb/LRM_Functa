import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import compute_metrics, compute_regression_metrics


# -------------------------
# Unified evaluation
# -------------------------
def evaluate(
    model,
    loader,
    criterion,
    device,
    task="classification",  # "classification" | "echo"
):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            
            # ---- support both formats ----
            if isinstance(batch, dict):
                X = batch["images"]
                y = batch["label"]
            else:
                X = batch[0]
                y = batch[1][:, 0]

            X = X.float().to(device)
            y = y.to(device)

            outputs = model(X).flatten()

            if torch.isnan(outputs).any():
                continue

            loss = criterion(outputs, y.float())
            running_loss += loss.item()

            all_targets.append(y.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    # ---- aggregate ----
    targets = np.concatenate(all_targets)
    outputs = np.concatenate(all_outputs)

    # ---- choose metrics ----
    if task == "echo":
        metrics = compute_regression_metrics(targets, outputs)
    else:
        metrics = compute_metrics(targets, outputs)

    metrics["loss"] = running_loss / len(loader)

    return metrics, outputs


# -------------------------
# Save predictions
# -------------------------
def evaluate_and_save(
    model,
    loader,
    criterion,
    device,
    save_path,
    task="classification",
):
    model.eval()

    all_targets, all_outputs = [], []
    all_probs, all_preds, all_ids = [], [], []

    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Saving predictions"):

            X = batch["images"].float().to(device)
            y = batch["label"].to(device)

            outputs = model(X).flatten()

            loss = criterion(outputs, y.float())
            running_loss += loss.item()

            all_targets.append(y.cpu())
            all_outputs.append(outputs.cpu())

            if task != "echo":
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())

            if "id" in batch:
                all_ids.append(batch["id"])

    # ---- stack ----
    targets = torch.cat(all_targets).numpy()
    outputs = torch.cat(all_outputs).numpy()

    # ---- metrics ----
    if task == "echo":
        metrics = compute_regression_metrics(targets, outputs)
    else:
        metrics = compute_metrics(targets, outputs)

    metrics["loss"] = running_loss / len(loader)

    # ---- ids ----
    try:
        ids = torch.cat(all_ids).numpy()
    except:
        ids = np.arange(len(targets))

    # ---- save dataframe ----
    if task == "echo":
        df = pd.DataFrame({
            "id": ids,
            "target": targets,
            "prediction": outputs,
        })
    else:
        probs = torch.cat(all_probs).numpy()
        preds = torch.cat(all_preds).numpy()

        df = pd.DataFrame({
            "id": ids,
            "target": targets,
            "prediction": preds,
            "probability": probs,
        })

    df.to_csv(save_path, index=False)

    return metrics, outputs
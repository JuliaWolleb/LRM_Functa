"""
Clean 5-Fold CV for B-line classification on the lung ultrasound videos.
Uses dataset.get_dataset(config) and splits in training script only
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd

import torch
import wandb

from torch.utils.data import DataLoader, Subset
from torchvision import models as torchmodels

from sklearn.model_selection import KFold, train_test_split

import ml_collections

sys.path.append('.')
sys.path.append('..')

from deepchest.utilities import config_utils, utils
from deepchest.dataset_loading import dataset
from deepchest.evaluation.metrics import compute_metrics
from deepchest.evaluation.model_evaluation import model_evaluation
# ============================================================
# CREATE FIXED RANDOM TEST SET
# ============================================================

def create_fixed_test_split(full_dataset, config):

    print("\nCreating fixed random test set of size", config.fixed_test_size)

    np.random.seed(config.seed)

    all_indices = np.arange(len(full_dataset))

    test_indices = np.random.choice(
        all_indices,
        size=config.fixed_test_size,
        replace=False
    )

    train_val_indices = np.setdiff1d(
        all_indices,
        test_indices
    )

    print("Train/Val size:", len(train_val_indices))
    print("Test size:", len(test_indices))

    return train_val_indices, test_indices


# ============================================================
# CONFIG
# ============================================================

def get_config():

    config = ml_collections.ConfigDict()

    config.seed = 30

    config.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config.wandb = False

    config.save_dir = "./saved_runs"
    config.run_name = "original"

    config.datasetoption = "lung"
    config.pathology = "bline"

    config.learning_rate = 0.001
    config.weight_decay = 0.0

    config.batch_size = 10
    config.num_workers = 0

    config.nb_epochs = 20
    config.fixed_test_size = 40
    config.num_folds = 5

    config.pos_weight = torch.tensor([1.0])

    config.modelname = "pocovid"

    config.clip = 120

    config.labels_file = "./data/lung/Clips_allinformation.csv"
    config.videos_directory = "./data/lung/videos"

    return config


# ============================================================
# MODEL
# ============================================================

class Video3DCNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.base_model = torchmodels.video.r3d_18(pretrained=True)

        self.base_model.fc = torch.nn.Linear(
            self.base_model.fc.in_features,
            1
        )

    def forward(self, x):

        return self.base_model(x)


# ============================================================
# CSV SAVE FUNCTION
# ============================================================

def save_results_csv(results, csv_path):

    df = pd.DataFrame(results)

    if len(df) == 0:
        return

    # compute mean and std safely
    mean_row = df.mean(numeric_only=True).to_dict()
    std_row = df.std(numeric_only=True).to_dict()

    mean_row["fold"] = "mean"
    std_row["fold"] = "std"

    df_final = pd.concat(
        [
            df,
            pd.DataFrame([mean_row]),
            pd.DataFrame([std_row])
        ],
        ignore_index=True
    )

    df_final.to_csv(csv_path, index=False)

# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
        fold,
        train_loader,
        val_loader,
        test_loader,
        config,
        save_dir
):

    print(f"\n========== Fold {fold} ==========")

    run = None

    model = Video3DCNN().to(config.device)

    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.98
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=config.pos_weight.to(config.device)
    )

    best_auc = -1

    best_checkpoint = save_dir / f"checkpoint_fold{fold}_best.pth"
    last_checkpoint = save_dir / f"checkpoint_fold{fold}_last.pth"


    for epoch in range(config.nb_epochs):
        print('Epoch', epoch)
        model.train()

        targets = []
        preds = []

        for batch in train_loader:

            images = batch["images"].to(config.device)
            labels = batch["label"].float().to(config.device)

            optimizer.zero_grad()

            outputs = model(images).flatten()

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            targets.extend(labels.cpu().numpy())
            preds.extend(outputs.detach().cpu().numpy())

        scheduler.step()

        train_metrics = compute_metrics(targets, preds)

        val_metrics = model_evaluation(
            model,
            val_loader,
            criterion,
            config.device
        )[0]

        test_metrics = model_evaluation(
            model,
            test_loader,
            criterion,
            config.device
        )[0]



        

        print("Val ROC AUC:", val_metrics["roc_auc"])
        print("test ROC AUC:", test_metrics["roc_auc"])


        if val_metrics["roc_auc"] > best_auc:

            best_auc = val_metrics["roc_auc"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                },
                best_checkpoint
            )

            print("Saved best checkpoint")


    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        },
        last_checkpoint
    )


    checkpoint = torch.load(best_checkpoint)

    model.load_state_dict(checkpoint["model"])

    test_metrics = model_evaluation(
        model,
        test_loader,
        criterion,
        config.device
    )[0]

    print("Test metrics:", test_metrics)


    return test_metrics


# ============================================================
# MAIN
# ============================================================
def main(config):

    utils.set_seed(config.seed)

    save_dir = pathlib.Path(config.save_dir) / config.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "original_results.csv"

    # ========================================================
    # LOAD FULL DATASET
    # ========================================================

    full_dataset = dataset.get_dataset(config)

    print("Dataset size:", len(full_dataset))

    # ========================================================
    # CREATE FIXED TEST SET
    # ========================================================

    train_val_indices, test_indices = create_fixed_test_split(
        full_dataset,
        config
    )

    test_subset = Subset(full_dataset, test_indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # ========================================================
    # K-FOLD ON TRAIN_VAL ONLY
    # ========================================================

    kfold = KFold(
        n_splits=config.num_folds,
        shuffle=True,
        random_state=config.seed
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(
            kfold.split(train_val_indices)):

        print(f"\nPreparing fold {fold}")

        train_indices_fold = train_val_indices[train_idx]
        val_indices_fold = train_val_indices[val_idx]

        train_subset = Subset(full_dataset, train_indices_fold)
        val_subset = Subset(full_dataset, val_indices_fold)

        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        test_metrics = train_fold(
            fold,
            train_loader,
            val_loader,
            test_loader,   # SAME TEST SET EVERY FOLD
            config,
            save_dir
        )

        row = {"fold": fold}
        row.update(test_metrics)

        results.append(row)

        save_results_csv(results, csv_path)

        print("CSV updated:", csv_path)

    print("\nTraining complete")
    print("Results saved to:", csv_path)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    wandb.login()

    config = config_utils.parse_cli_overides(get_config())

    main(config)

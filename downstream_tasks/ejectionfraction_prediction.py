"""
DeepChest Echo Training Script — Clean 5-Fold CV with Hold-Out Test

Features:
• 5-fold CV on training portion
• fixed hold-out test set of 200 samples
• progressive CSV logging
• safe checkpointing (no overwrite)
• wandb logging per fold
• automatic mean/std computation

Author: You + Cleaned version
"""

import os
import sys
import pathlib
import yaml
import csv
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
from deepchest.dataset_loading import dataset_echonet
from deepchest.evaluation.metrics import compute_metrics_echo
from deepchest.evaluation.model_evaluation import model_evaluation_echo


# ============================================================
# Config
# ============================================================

def get_config():

    config = ml_collections.ConfigDict()

    cwd = os.getcwd()
    config.base_dir = cwd

    config.wandb = True
    config.save_dir = "./saved_runs_echo"
    config.run_name = "original"
    config.pathology = 'ejection_fraction'



    config.seed = 42
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.learning_rate = 0.001
    config.weight_decay = 0.0
    config.batch_size = 16

    config.nb_epochs = 40

    config.num_workers = 4

    config.clip = 60
    config.period = 5

    config.modelname = "pocovid"

    config.test_holdout_size = 500
    config.num_folds = 5

    return config


# ============================================================
# Model
# ============================================================

class Video3DCNN(torch.nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        self.base_model = torchmodels.video.r3d_18(pretrained=True)

        self.base_model.fc = torch.nn.Linear(
            self.base_model.fc.in_features,
            num_classes
        )

    def forward(self, x):

        return self.base_model(x)


# ============================================================
# CSV Logger
# ============================================================

def save_results_csv(results, csv_path):
    df = pd.DataFrame(results)

    # Compute mean and std
    mean_row = df.mean(numeric_only=True).to_frame().T
    std_row = df.std(numeric_only=True).to_frame().T

    # Add fold labels
    mean_row["fold"] = "mean"
    std_row["fold"] = "std"

    # Concatenate with original dataframe
    df_final = pd.concat([df, mean_row, std_row], ignore_index=True)

    # Save to CSV
    df_final.to_csv(csv_path, index=False)
# ============================================================
# Training Function
# ============================================================

def train_one_fold(
    fold,
    train_loader,
    val_loader,
    test_loader,
    config,
    save_dir
):

    print(f"\n========== Fold {fold} ==========")

    wandb_run = None

    if config.wandb:

        wandb_run = wandb.init(
            project="basis",
            name=f"original_fold_{fold}",
            reinit=True
        )

    model = Video3DCNN(1).to(config.device)

    if config.wandb:
        wandb.watch(model)

    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.98
    )

    criterion = torch.nn.MSELoss().to(config.device)

    best_mae = float("inf")

    best_checkpoint_path = save_dir / f"checkpoint_fold{fold}_best.pth"
    last_checkpoint_path = save_dir / f"checkpoint_fold{fold}_last.pth"

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(config.nb_epochs):

        model.train()

        targets = []
        preds = []

        for X, target in train_loader:

            X = X.float().to(config.device)
            target = target[:, 0].float().to(config.device)

            optimizer.zero_grad()

            output = model(X).flatten()

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            targets.extend(target.cpu().numpy())
            preds.extend(output.detach().cpu().numpy())

        scheduler.step()

        train_metrics = compute_metrics_echo(targets, preds)

        if config.wandb:
            wandb.log(
                utils.prefix_dict(train_metrics, "train/"),
                step=epoch
            )

        # ==========================
        # Validation
        # ==========================

        val_metrics = model_evaluation_echo(
            model,
            val_loader,
            criterion,
            config.device
        )[0]

        if config.wandb:
            wandb.log(
                utils.prefix_dict(val_metrics, "valid/"),
                step=epoch
            )

        print(f"Epoch {epoch} | Val MAE: {val_metrics['mae']}")

        if val_metrics["mae"] < best_mae:

            best_mae = val_metrics["mae"]

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, best_checkpoint_path)

            print("Saved best checkpoint")

    # save last checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, last_checkpoint_path)

    # ========================================================
    # Test best model
    # ========================================================

    checkpoint = torch.load(best_checkpoint_path)

    model.load_state_dict(checkpoint["model"])

    test_metrics = model_evaluation_echo(
        model,
        test_loader,
        criterion,
        config.device
    )[0]

    print("Test metrics:", test_metrics)

    if config.wandb:

        wandb.log(
            utils.prefix_dict(test_metrics, "test/")
        )

        wandb_run.finish()

    return test_metrics


# ============================================================
# Main
# ============================================================

def main(config):

    utils.set_seed(config.seed)

    save_dir = pathlib.Path(config.save_dir) / config.run_name

    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "echo_results.csv"

    print("Loading dataset...")

    
    full_dataset = dataset_echonet.Echo(external_test_location = './data/EchoNet-Dynamic/videos', split="EXTERNAL_TEST", period=5, length=config.clip  )
   #  full_dataset = dataset_echonet.Echo(external_test_location = './reconstructions/ ... /videos', split="EXTERNAL_TEST", period=5, length=config.clip  )  #put the path to your reconstructed videos here



    print('full length', len(full_dataset))
    
    indices = list(range(len(full_dataset)))

    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_holdout_size,
        random_state=42,
        shuffle=True
    )

    cv_dataset = Subset(full_dataset, trainval_idx)
    test_dataset = Subset(full_dataset, test_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    print("CV size:", len(cv_dataset))
    print("Test size:", len(test_dataset))

    kfold = KFold(
        n_splits=config.num_folds,
        shuffle=True,
        random_state=42
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(cv_dataset)):
      
        train_subset = Subset(cv_dataset, train_idx)
        val_subset = Subset(cv_dataset, val_idx)

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

        test_metrics = train_one_fold(
            fold,
            train_loader,
            val_loader,
            test_loader,
            config,
            save_dir
        )

        row = {"fold": fold}

        row.update(test_metrics)

        results.append(row)

        save_results_csv(results, csv_path)

        print("CSV updated")

    print("\nTraining complete")
    print("Results saved to:", csv_path)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    wandb.login()

    config = config_utils.parse_cli_overides(get_config())

    main(config)

import os
import argparse
import random
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from model.efficientnet_lstm import efficientnet_lstm
from model.x3d_model import x3d_model
from model.mvit_v2 import MViT_v2_S
from preprocessing.dataset import FaceForensicsDataset
from utils.logger import setup_logger
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.lr_scheduler import lr_scheduler


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg):
    name = cfg["model"]["name"].lower()
    if "efficientnet_lstm" in name:
        return efficientnet_lstm()
    elif "x3d" in name:
        return x3d_model()
    elif "mvit_v2_s" in name:
        return MViT_v2_S()
    else:
        raise ValueError(f"[WARN] Unknown model name: {name}")


def get_args():
    parser = argparse.ArgumentParser(description="Train deepfake classification model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("-ckpt", "--ckpt", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def train():
    args = get_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger
    save_path = os.path.join("checkpoints", cfg["model"]["name"])
    os.makedirs(save_path, exist_ok=True)
    logger = setup_logger(save_path, log_name=f"{cfg['model']['name']}_train")

    # Build Model
    model = build_model(cfg).to(device)

    # Dataset & Dataloader
    train_ds = FaceForensicsDataset(cfg["data"]["root_dir"], phase="train")
    val_ds = FaceForensicsDataset(cfg["data"]["root_dir"], phase="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )

    # Optimizer / Scheduler / Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = lr_scheduler(
        optimizer,
        warmup_epochs=cfg["train"]["warmup_epochs"],
        total_epochs=cfg["train"]["epochs"],
        max_lr=cfg["train"]["lr"],
        min_lr=cfg["train"]["min_lr"]
    )

    # Weighted BCE
    weights = torch.tensor([2.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Resume Checkpoint
    start_epoch = 0
    if args.ckpt is not None and os.path.exists(args.ckpt):
        model, optimizer, start_epoch = load_checkpoint(args.ckpt, model, optimizer, device)
        scheduler.last_epoch = start_epoch - 1

    num_epochs = cfg["train"]["epochs"]

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = []
        all_y_true, all_y_prob = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        for batch in pbar:
            videos = batch["clip"].to(device)
            labels = batch["label"].long().to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            train_losses.append(loss.item())

            all_y_true.extend(labels.cpu().numpy())
            all_y_prob.extend(probs)

            pbar.set_postfix({"loss": f"{np.mean(train_losses):.4f}"})

        scheduler.step()

        avg_train_loss = np.mean(train_losses)
        try:
            preds = (np.array(all_y_prob) > 0.5).astype(int)
            train_f1 = f1_score(all_y_true, preds)
            train_roc = roc_auc_score(all_y_true, all_y_prob)
            train_pr = average_precision_score(all_y_true, all_y_prob)
        except ValueError:
            train_f1 = train_roc = train_pr = 0.0

        # Validation 
        model.eval()
        val_losses = []
        val_true, val_prob = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", ncols=100):
                videos = batch["clip"].to(device)
                labels = batch["label"].long().to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                val_losses.append(loss.item())
                val_true.extend(labels.cpu().numpy())
                val_prob.extend(probs)

        avg_val_loss = np.mean(val_losses)
        try:
            val_preds = (np.array(val_prob) > 0.5).astype(int)
            val_f1 = f1_score(val_true, val_preds)
            val_roc = roc_auc_score(val_true, val_prob)
            val_pr = average_precision_score(val_true, val_prob)
        except ValueError:
            val_f1 = val_roc = val_pr = 0.0

        # Logging & Save
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | train_f1={train_f1:.4f} | "
            f"train_auc={train_roc:.4f} | train_pr_auc={train_pr:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_f1={val_f1:.4f} | "
            f"val_auc={val_roc:.4f} | val_pr_auc={val_pr:.4f}"
        )

        print(
            f"## Epoch [{epoch+1}] | "
            f"train_loss={avg_train_loss:.4f} | train_f1={train_f1:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_f1={val_f1:.4f} | "
            f"val_auc={val_roc:.4f} | val_pr_auc={val_pr:.4f}"
        )

        # Save checkpoint
        save_checkpoint(model, optimizer, save_path, epoch)


if __name__ == "__main__":
    train()
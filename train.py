"""
Unified training script for Camouflaged Object Detection.

Supports all model architectures:
  - UNet, UNet_B3, UNet_BEM
  - UNetPP, UNetPP_B3, UNetPP_BEM
  - UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM

Features:
  - Combined loss: Dice + Focal + Boundary (with warmup)
  - Differential learning rates (encoder vs decoder/BEM)
  - Encoder warmup (optional freeze)
  - Boundary loss warmup strategy
  - Cosine annealing scheduler
  - Mixed precision training (AMP)
  - CSV logging + training curve plots
  - Checkpoint resume
"""

import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from datetime import datetime

from datasets.mhcd_dataset import MHCDDataset
from models import (
    UNet, UNet_B3, UNet_BEM,
    UNetPP, UNetPP_B3, UNetPP_BEM,
    UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM,
    BoundaryEnhancementModule,
)
from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.fweighted_measure import fw_measure
from logger_utils import setup_logger


# ===================== Model Registry =====================

MODEL_REGISTRY = {
    # UNet
    "UNet":          UNet,
    "UNet_B3":       UNet_B3,
    "UNet_BEM":      UNet_BEM,
    # UNet++
    "UNetPP":        UNetPP,
    "UNetPP_B3":     UNetPP_B3,
    "UNetPP_BEM":    UNetPP_BEM,
    # UNet3+
    "UNet3Plus":     UNet3Plus,
    "UNet3Plus_B3":  UNet3Plus_B3,
    "UNet3Plus_BEM": UNet3Plus_BEM,
}

# Models that return (mask, boundary) tuple
BEM_MODELS = {"UNet_BEM", "UNetPP_BEM", "UNet3Plus_BEM"}


# ===================== Loss Functions =====================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * torch.pow((1 - p_t), self.gamma)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (focal_weight * bce_loss).mean()


class SegmentationLoss(nn.Module):
    """
    Combined loss for camouflaged object detection.
    Components: BCE + Dice + IoU + Focal + Boundary (L1 + BCE)
    """
    def __init__(self, lambda_bce=0.0, lambda_dice=0.35, lambda_iou=0.0,
                 lambda_focal=0.35, lambda_boundary=0.3,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_iou = lambda_iou
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary

        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def dice_loss(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()
        return 1 - (2. * intersection + smooth) / (union + smooth)

    def iou_loss(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum() - intersection
        return 1 - (intersection + smooth) / (union + smooth)

    def boundary_loss(self, boundary_pred, boundary_target):
        pred_prob = torch.sigmoid(boundary_pred)
        l1_loss = F.l1_loss(pred_prob, boundary_target)
        bce_loss = F.binary_cross_entropy_with_logits(boundary_pred, boundary_target)
        return l1_loss + bce_loss

    def forward(self, pred, target, boundary_pred=None, boundary_target=None):
        total_loss = 0.0

        if self.lambda_bce > 0:
            total_loss += self.lambda_bce * self.bce(pred, target)
        if self.lambda_dice > 0:
            total_loss += self.lambda_dice * self.dice_loss(pred, target)
        if self.lambda_iou > 0:
            total_loss += self.lambda_iou * self.iou_loss(pred, target)
        if self.lambda_focal > 0:
            total_loss += self.lambda_focal * self.focal(pred, target)

        if boundary_pred is not None and boundary_target is not None and self.lambda_boundary > 0:
            total_loss += self.lambda_boundary * self.boundary_loss(boundary_pred, boundary_target)

        return total_loss


# ===================== Utility =====================

def mae_metric(pred, target):
    """Mean Absolute Error metric"""
    return torch.mean(torch.abs(torch.sigmoid(pred) - target))


def extract_boundary_sobel(masks, device):
    """
    Extract boundary maps from ground truth masks using Sobel filters.
    Args:
        masks: (B, 1, H, W) binary masks
        device: torch device
    Returns:
        boundary: (B, 1, H, W) normalized boundary maps
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    gx = F.conv2d(masks, sobel_x, padding=1)
    gy = F.conv2d(masks, sobel_y, padding=1)
    boundary = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # Normalize per-sample
    B = boundary.shape[0]
    for i in range(B):
        bmin = boundary[i].min()
        bmax = boundary[i].max()
        boundary[i] = (boundary[i] - bmin) / (bmax - bmin + 1e-8)

    return boundary


def get_encoder_params(model):
    """Get encoder parameter names depending on model architecture."""
    encoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # smp-based models: base.encoder.*
        # UNet3Plus models: encoder.*
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            other_params.append(param)

    return encoder_params, other_params


def freeze_encoder(model, freeze=True):
    """Freeze or unfreeze the encoder."""
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = not freeze


# ===================== Training =====================

def train_epoch(model, loader, optimizer, criterion, scaler, device, args, epoch):
    model.train()

    # Freeze encoder during warmup
    if args.warmup_epochs > 0:
        if epoch <= args.warmup_epochs:
            freeze_encoder(model, freeze=True)
        elif epoch == args.warmup_epochs + 1:
            freeze_encoder(model, freeze=False)

    is_bem = args.model in BEM_MODELS
    use_boundary = is_bem and (epoch > args.warmup_boundary_epochs)

    total_loss = 0.0
    num_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            output = model(images)

            if is_bem and isinstance(output, tuple):
                mask_pred, boundary_pred = output
            else:
                mask_pred = output
                boundary_pred = None

            if use_boundary and boundary_pred is not None:
                boundary_target = extract_boundary_sobel(masks, device)
                loss = criterion(mask_pred, masks, boundary_pred, boundary_target)
            else:
                loss = criterion(mask_pred, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ===================== Validation =====================

@torch.no_grad()
def validate(model, loader, criterion, device, args):
    model.eval()

    is_bem = args.model in BEM_MODELS

    metrics = {"loss": 0.0, "S": 0.0, "E": 0.0, "Fw": 0.0, "MAE": 0.0}
    num_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        output = model(images)

        if is_bem and isinstance(output, tuple):
            mask_pred, boundary_pred = output
            boundary_target = extract_boundary_sobel(masks, device)
            loss = criterion(mask_pred, masks, boundary_pred, boundary_target)
        else:
            mask_pred = output
            loss = criterion(mask_pred, masks)

        pred_probs = torch.sigmoid(mask_pred)
        batch_size = images.shape[0]

        for i in range(batch_size):
            metrics["S"] += s_measure(pred_probs[i], masks[i]).item()
            metrics["E"] += e_measure(pred_probs[i], masks[i]).item()
            metrics["Fw"] += fw_measure(pred_probs[i], masks[i]).item()
            metrics["MAE"] += mae_metric(mask_pred[i:i+1], masks[i:i+1]).item()

        metrics["loss"] += loss.item() * batch_size
        num_samples += batch_size

    for key in metrics:
        metrics[key] /= max(num_samples, 1)

    return metrics


# ===================== Visualization =====================

def plot_training_curves(train_losses, val_losses, val_metrics, save_dir):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(epochs, train_losses, label='Train', marker='o', markersize=3)
    axes[0, 0].plot(epochs, val_losses, label='Val', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # S-measure
    axes[0, 1].plot(epochs, [m["S"] for m in val_metrics], color='green', marker='o', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('S-measure')
    axes[0, 1].set_title('S-measure')
    axes[0, 1].grid(True)

    # E-measure + Fw-measure
    axes[1, 0].plot(epochs, [m["E"] for m in val_metrics], label='E-measure', marker='o', markersize=3)
    axes[1, 0].plot(epochs, [m["Fw"] for m in val_metrics], label='Fw-measure', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('E-measure & Fw-measure')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # MAE
    axes[1, 1].plot(epochs, [m["MAE"] for m in val_metrics], color='red', marker='o', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MAE')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


# ===================== Main =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Train COD models")

    # Model
    parser.add_argument('--model', type=str, default='UNetPP_BEM',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')

    # Dataset
    parser.add_argument('--root', type=str, default='../MHCD_seg',
                        help='Dataset root directory')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')

    # Training
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)

    # Optimizer
    parser.add_argument('--lr_encoder', type=float, default=1e-5,
                        help='Learning rate for encoder (pretrained)')
    parser.add_argument('--lr_decoder', type=float, default=1e-4,
                        help='Learning rate for decoder / BEM')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Loss weights
    parser.add_argument('--lambda_bce', type=float, default=0.0)
    parser.add_argument('--lambda_dice', type=float, default=0.35)
    parser.add_argument('--lambda_iou', type=float, default=0.0)
    parser.add_argument('--lambda_focal', type=float, default=0.35)
    parser.add_argument('--lambda_boundary', type=float, default=0.3)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # Warmup
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Freeze encoder for first N epochs')
    parser.add_argument('--warmup_boundary_epochs', type=int, default=30,
                        help='No boundary loss for first N epochs (BEM models only)')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'])

    # Misc
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (auto-generated if not set)')

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log directory
    if args.log_dir is None:
        args.log_dir = f"logs/{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(args.log_dir, "train.log")

    # --- Log config ---
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k:25s}: {v}")
    logger.info(f"  {'device':25s}: {device}")
    logger.info("=" * 80)

    # --- Dataset ---
    logger.info("Loading datasets...")
    train_dataset = MHCDDataset(args.root, "train", args.img_size, logger=logger)
    val_dataset = MHCDDataset(args.root, "val", args.img_size, logger=logger)
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    logger.info(f"Creating model: {args.model}")
    model = MODEL_REGISTRY[args.model](n_classes=1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # --- Optimizer with differential LR ---
    encoder_params, other_params = get_encoder_params(model)
    param_groups = []
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': args.lr_encoder, 'name': 'encoder'})
        logger.info(f"  Encoder params: {len(encoder_params)} tensors, LR={args.lr_encoder}")
    if other_params:
        param_groups.append({'params': other_params, 'lr': args.lr_decoder, 'name': 'decoder'})
        logger.info(f"  Decoder/BEM params: {len(other_params)} tensors, LR={args.lr_decoder}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # --- Loss ---
    criterion = SegmentationLoss(
        lambda_bce=args.lambda_bce, lambda_dice=args.lambda_dice,
        lambda_iou=args.lambda_iou, lambda_focal=args.lambda_focal,
        lambda_boundary=args.lambda_boundary,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
    )

    # --- Scheduler ---
    if args.scheduler == 'cosine':
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    scaler = GradScaler(enabled=(device == 'cuda'))

    # --- Resume ---
    start_epoch = 1
    best_s_measure = 0.0
    train_losses = []
    val_losses = []
    val_metrics_history = []

    resume_path = args.resume or os.path.join(args.log_dir, "last.pth")
    if os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_s_measure = ckpt.get("best_s_measure", 0.0)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        val_metrics_history = ckpt.get("val_metrics", [])
        logger.info(f"Resumed at epoch {start_epoch}, best S={best_s_measure:.4f}")

    # --- CSV logging ---
    csv_path = os.path.join(args.log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                     "S_measure", "E_measure", "Fw_measure", "MAE"])

    # --- Training loop ---
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*60} Epoch {epoch}/{args.epochs} {'='*60}")

        if epoch <= args.warmup_epochs:
            logger.info("[WARMUP] Encoder frozen")

        if args.model in BEM_MODELS and epoch <= args.warmup_boundary_epochs:
            logger.info(f"[BOUNDARY WARMUP] Boundary loss disabled (epoch <= {args.warmup_boundary_epochs})")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, args, epoch)
        train_losses.append(train_loss)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, args)
        val_losses.append(val_metrics["loss"])
        val_metrics_history.append(val_metrics)

        # Log
        logger.info(f"[TRAIN] Loss: {train_loss:.4f}")
        logger.info(f"[VAL]   Loss: {val_metrics['loss']:.4f} | "
                     f"S: {val_metrics['S']:.4f} | "
                     f"E: {val_metrics['E']:.4f} | "
                     f"Fw: {val_metrics['Fw']:.4f} | "
                     f"MAE: {val_metrics['MAE']:.4f}")

        # CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_metrics["loss"],
                val_metrics["S"], val_metrics["E"], val_metrics["Fw"], val_metrics["MAE"]
            ])

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_s_measure": best_s_measure,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics_history,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.log_dir, "last.pth"))

        # Best model
        if val_metrics["S"] > best_s_measure:
            best_s_measure = val_metrics["S"]
            torch.save(model.state_dict(), os.path.join(args.log_dir, "best_s_measure.pth"))
            logger.info(f"** NEW BEST S-measure: {best_s_measure:.4f} **")

        # Scheduler step
        if scheduler is not None:
            if args.scheduler == 'cosine':
                for _ in range(len(train_loader)):
                    scheduler.step()
            else:
                scheduler.step()

        # Plot
        if epoch % 5 == 0 or epoch == args.epochs:
            plot_training_curves(train_losses, val_losses, val_metrics_history, args.log_dir)

    # --- Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best S-measure: {best_s_measure:.4f}")
    logger.info(f"Logs: {args.log_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

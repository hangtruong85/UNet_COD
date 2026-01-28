"""
Evaluation script for Camouflaged Object Detection models.

Supports all model architectures:
  - UNet, UNet_B3, UNet_BEM
  - UNetPP, UNetPP_B3, UNetPP_BEM
  - UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM

Usage:
  python evaluate.py                         # Evaluate single model (edit config below)
  python evaluate.py --compare               # Compare multiple models
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from datasets.mhcd_dataset import MHCDDataset
from models import (
    UNet, UNet_B3, UNet_BEM,
    UNetPP, UNetPP_B3, UNetPP_BEM,
    UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM,
)
from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.f_measure_paper import f_measure
from metrics.fweighted_measure import fw_measure
from logger_utils import setup_logger


# =========================================================
# Model Registry
# =========================================================

MODEL_REGISTRY = {
    "UNet":          UNet,
    "UNet_B3":       UNet_B3,
    "UNet_BEM":      UNet_BEM,
    "UNetPP":        UNetPP,
    "UNetPP_B3":     UNetPP_B3,
    "UNetPP_BEM":    UNetPP_BEM,
    "UNet3Plus":     UNet3Plus,
    "UNet3Plus_B3":  UNet3Plus_B3,
    "UNet3Plus_BEM": UNet3Plus_BEM,
}

BEM_MODELS = {"UNet_BEM", "UNetPP_BEM", "UNet3Plus_BEM"}


# =========================================================
# Configuration
# =========================================================

class EvalConfig:
    def __init__(self):
        self.model_name = "UNetPP_BEM"
        self.ckpt_path = "logs/UNetPP_BEM/best_s_measure.pth"
        self.root = "../MHCD_seg"
        self.split = "test"
        self.img_size = 640
        self.batch_size = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = f"eval_results/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)


# =========================================================
# Checkpoint Loading
# =========================================================

def load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "unknown")
        best_metric = ckpt.get("best_s_measure", "unknown")
        print(f"  Loaded from epoch {epoch}, best S-measure: {best_metric}")
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        print(f"  Loaded state_dict")
    else:
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")

    return model


# =========================================================
# Metrics Accumulator
# =========================================================

class MetricsAccumulator:
    def __init__(self):
        self.metrics = {"S": [], "E": [], "F": [], "Fw": [], "MAE": []}

    def update(self, pred, target):
        self.metrics["S"].append(s_measure(pred, target).item())
        self.metrics["E"].append(e_measure(pred, target).item())
        self.metrics["F"].append(f_measure(pred, target).item())
        self.metrics["Fw"].append(fw_measure(pred, target).item())
        self.metrics["MAE"].append(torch.abs(pred - target).mean().item())

    def get_summary(self):
        summary = {}
        for key, values in self.metrics.items():
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        return summary

    def get_detailed(self):
        return self.metrics


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device, model_name, logger):
    model.eval()
    accumulator = MetricsAccumulator()
    is_bem = model_name in BEM_MODELS

    logger.info("Starting evaluation...")
    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        output = model(images)

        # BEM models return (mask, boundary) tuple
        if is_bem and isinstance(output, tuple):
            pred_logits = output[0]
        else:
            pred_logits = output

        pred_probs = torch.sigmoid(pred_logits)

        for i in range(images.shape[0]):
            accumulator.update(pred_probs[i], masks[i])

        current = accumulator.get_summary()
        pbar.set_postfix({'S': f"{current['S_mean']:.4f}", 'MAE': f"{current['MAE_mean']:.4f}"})

    return accumulator


# =========================================================
# Results Display / Save
# =========================================================

def display_results(summary, logger):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric in ['S', 'E', 'F', 'Fw', 'MAE']:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        print(f"{metric:8s}: {mean:.4f}")
        logger.info(f"{metric:12s}: {mean:.4f} +/- {std:.4f}")
    print("=" * 60 + "\n")


def save_results(summary, detailed, config):
    summary_path = os.path.join(config.save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    detailed_path = os.path.join(config.save_dir, "detailed.json")
    with open(detailed_path, 'w') as f:
        json.dump(detailed, f, indent=4)

    config_dict = {
        "model_name": config.model_name,
        "checkpoint": config.ckpt_path,
        "dataset_root": config.root,
        "split": config.split,
        "img_size": config.img_size,
        "batch_size": config.batch_size,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = os.path.join(config.save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    print(f"Results saved to: {config.save_dir}")


# =========================================================
# Main
# =========================================================

def main():
    config = EvalConfig()
    logger = setup_logger(config.save_dir, "evaluation.log")

    logger.info("=" * 80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name} | Checkpoint: {config.ckpt_path}")
    logger.info(f"Dataset: {config.root}/{config.split} | Image size: {config.img_size}")
    logger.info("=" * 80)

    model = MODEL_REGISTRY[config.model_name](n_classes=1).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params:,}")

    model = load_checkpoint(model, config.ckpt_path, config.device)

    dataset = MHCDDataset(config.root, config.split, config.img_size, logger=logger)
    logger.info(f"Samples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    accumulator = evaluate_model(model, dataloader, config.device, config.model_name, logger)

    summary = accumulator.get_summary()
    detailed = accumulator.get_detailed()

    display_results(summary, logger)
    save_results(summary, detailed, config)


def compare_models():
    """Compare multiple models."""
    models_to_compare = [
        ("UNet_B3",       "logs/UNet_B3/best_s_measure.pth"),
        ("UNet_BEM",      "logs/UNet_BEM/best_s_measure.pth"),
        ("UNetPP_B3",     "logs/UNetPP_B3/best_s_measure.pth"),
        ("UNetPP_BEM",    "logs/UNetPP_BEM/best_s_measure.pth"),
        ("UNet3Plus_B3",  "logs/UNet3Plus_B3/best_s_measure.pth"),
        ("UNet3Plus_BEM", "logs/UNet3Plus_BEM/best_s_measure.pth"),
    ]

    config = EvalConfig()
    os.makedirs("eval_results/comparison", exist_ok=True)
    logger = setup_logger("eval_results/comparison", "comparison.log")

    dataset = MHCDDataset(config.root, config.split, config.img_size, logger=logger)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=False, num_workers=4)

    results_comparison = {}

    for model_name, ckpt_path in models_to_compare:
        if not os.path.exists(ckpt_path):
            print(f"Skipping {model_name}: checkpoint not found")
            continue

        print(f"\nEvaluating: {model_name}")
        model = MODEL_REGISTRY[model_name](n_classes=1).to(config.device)
        model = load_checkpoint(model, ckpt_path, config.device)

        accumulator = evaluate_model(model, dataloader, config.device, model_name, logger)
        summary = accumulator.get_summary()
        results_comparison[model_name] = summary

        for key in ['S_mean', 'E_mean', 'Fw_mean', 'MAE_mean']:
            print(f"  {key:12s}: {summary[key]:.4f}")

    comparison_path = "eval_results/comparison/comparison_results.json"
    with open(comparison_path, 'w') as f:
        json.dump(results_comparison, f, indent=4)

    # Print table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'S':>10} {'E':>10} {'Fw':>10} {'MAE':>10}")
    print("-" * 60)
    for name, m in results_comparison.items():
        print(f"{name:<20} {m['S_mean']:>10.4f} {m['E_mean']:>10.4f} "
              f"{m['Fw_mean']:>10.4f} {m['MAE_mean']:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_models()
    else:
        main()

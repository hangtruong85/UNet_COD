"""
Visualization tool for comparing model predictions and feature maps.
Supports: UNet, UNet++, UNet3+ (with and without BEM)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from datasets.mhcd_dataset import MHCDDataset
from models import UNetPP_B3, UNetPP_BEM, UNet3Plus_B3, UNet3Plus_BEM


# ============================================================================
# Hook-based Feature Extractor
# ============================================================================

class FeatureExtractor:
    def __init__(self):
        self.features = {}
        self.hooks = []

    def register_hooks(self, model, layer_names):
        for name, module_path in layer_names.items():
            try:
                module = model
                for attr in module_path.split('.'):
                    module = getattr(module, attr)
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
            except Exception as e:
                print(f"Warning: Cannot hook {name} at {module_path}: {e}")

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# Attention Map Generator
# ============================================================================

class AttentionMapGenerator:
    @staticmethod
    def generate_cam(features, method='norm'):
        if method == 'mean':
            return features.mean(dim=1, keepdim=True)
        elif method == 'max':
            return features.max(dim=1, keepdim=True)[0]
        elif method == 'norm':
            return torch.norm(features, dim=1, keepdim=True)
        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def normalize_cam(cam):
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()


# ============================================================================
# Model Visualizer
# ============================================================================

class ModelVisualizer:
    def __init__(self, model, model_name, device='cuda'):
        self.model = model.to(device).eval()
        self.model_name = model_name
        self.device = device
        self.feature_extractor = FeatureExtractor()

    def visualize_sample(self, image, mask, save_path):
        image = image.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            pred_logits = output[0] if isinstance(output, tuple) else output
            pred_prob = torch.sigmoid(pred_logits)

        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        mask_np = mask[0, 0].cpu().numpy()
        pred_np = pred_prob[0, 0].cpu().numpy()

        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Input / GT / Prediction / Overlay / Error
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_np); ax1.set_title('Input'); ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask_np, cmap='gray'); ax2.set_title('Ground Truth'); ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred_np, cmap='gray'); ax3.set_title('Prediction'); ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(img_np); ax4.imshow(pred_np, cmap='jet', alpha=0.5)
        ax4.set_title('Overlay'); ax4.axis('off')

        ax5 = fig.add_subplot(gs[0, 4])
        error_map = np.abs(mask_np - pred_np)
        im = ax5.imshow(error_map, cmap='hot')
        ax5.set_title(f'Error (MAE={error_map.mean():.4f})'); ax5.axis('off')
        plt.colorbar(im, ax=ax5, fraction=0.046)

        # Row 2: Feature maps
        self._visualize_feature_maps(fig, gs, image, row=1)

        fig.suptitle(f'{self.model_name}', fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def _visualize_feature_maps(self, fig, gs, image, row):
        layer_paths = {}
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and 16 <= module.out_channels <= 512:
                conv_layers.append((name, module.out_channels))

        encoder_layers = [(n, c) for n, c in conv_layers if 'encoder' in n.lower()]
        decoder_layers = [(n, c) for n, c in conv_layers if 'decoder' in n.lower()]

        if encoder_layers:
            step = max(1, len(encoder_layers) // 3)
            for i, (name, _) in enumerate(encoder_layers[::step][:3]):
                layer_paths[f'enc_{i}'] = name
        if decoder_layers:
            layer_paths['dec_early'] = decoder_layers[0][0]
            if len(decoder_layers) >= 2:
                layer_paths['dec_late'] = decoder_layers[-1][0]

        if not layer_paths:
            ax = fig.add_subplot(gs[row, :])
            ax.text(0.5, 0.5, 'No feature layers found', ha='center', va='center')
            ax.axis('off')
            return

        self.feature_extractor.register_hooks(self.model, layer_paths)
        with torch.no_grad():
            _ = self.model(image)

        features = self.feature_extractor.features
        keys = list(features.keys())[:5]

        for i, key in enumerate(keys):
            ax = fig.add_subplot(gs[row, i])
            feat = features[key]
            cam = AttentionMapGenerator.generate_cam(feat, method='norm')
            cam_np = AttentionMapGenerator.normalize_cam(cam)
            im = ax.imshow(cam_np, cmap='jet')
            ax.set_title(f'{key}\n({feat.shape[1]}ch, {feat.shape[2]}x{feat.shape[3]})', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        self.feature_extractor.clear()
        self.feature_extractor.remove_hooks()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("FEATURE VISUALIZATION")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = "../MHCD_seg"
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)

    # Load models
    models_to_viz = {
        'UNetPP_B3':     UNetPP_B3(n_classes=1),
        'UNetPP_BEM':    UNetPP_BEM(n_classes=1),
        'UNet3Plus_B3':  UNet3Plus_B3(n_classes=1),
        'UNet3Plus_BEM': UNet3Plus_BEM(n_classes=1),
    }

    # Load weights if available
    for name, model in models_to_viz.items():
        ckpt_path = f'logs/{name}/best_s_measure.pth'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            print(f"Loaded weights for {name}")

    # Load dataset
    dataset = MHCDDataset(root, "val", img_size=352)
    print(f"Loaded {len(dataset)} validation samples")

    num_samples = min(5, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for idx in indices:
        image, mask = dataset[idx]
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        print(f"\nSample {idx}:")
        for name, model in models_to_viz.items():
            vis = ModelVisualizer(model, name, device)
            vis.visualize_sample(image, mask, f"{save_dir}/sample_{idx}_{name}.png")

    print(f"\nVisualizations saved to: {save_dir}/")


if __name__ == "__main__":
    main()

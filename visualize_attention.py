"""
Visualization tool for comparing UNet++ vs UNet3+ attention maps
Includes: Feature maps, DCN offsets, Attention visualization
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.mhcd_dataset import MHCDDataset
from models.unetpp_bem import UNetPP_B3, UNetPP_DCNv3_COD
from models.unet3plus_dcn import UNet3Plus_B3, UNet3Plus_DCNv3_COD


# ============================================================================
# Hook Manager for Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extract intermediate features using hooks"""
    def __init__(self):
        self.features = {}
        self.hooks = []
    
    def register_hooks(self, model, layer_names):
        """
        Register forward hooks to extract features
        Args:
            model: PyTorch model
            layer_names: dict mapping layer names to module paths
        """
        for name, module_path in layer_names.items():
            try:
                # Navigate to the module
                module = model
                for attr in module_path.split('.'):
                    if hasattr(module, attr):
                        module = getattr(module, attr)
                    else:
                        print(f"Warning: Cannot find {attr} in path {module_path}")
                        break
                else:
                    # Successfully found the module
                    hook = module.register_forward_hook(self._make_hook(name))
                    self.hooks.append(hook)
            except Exception as e:
                print(f"Warning: Failed to register hook for {name} at {module_path}: {e}")
    
    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
    
    def clear(self):
        """Clear stored features"""
        self.features = {}
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# DCN Offset Visualization
# ============================================================================

class DCNOffsetVisualizer:
    """Visualize deformable convolution offsets"""
    
    @staticmethod
    def extract_offsets(model, x):
        """
        Extract DCN offsets from model
        Returns dict of offsets at different layers
        """
        offsets = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, 'offset_conv'):
                    # DCNv1
                    offset = module.offset_conv(input[0])
                    offsets[name] = offset.detach()
                elif hasattr(module, 'offset_mask_conv'):
                    # DCNv2/v3
                    out = module.offset_mask_conv(input[0])
                    K = 9  # 3x3 kernel
                    offset = out[:, :2*K, :, :]
                    offsets[name] = offset.detach()
            return hook
        
        # Register hooks for DCN modules
        hooks = []
        for name, module in model.named_modules():
            if 'dcn' in name.lower() and ('DCNv1' in str(type(module)) or 
                                          'DCNv2' in str(type(module)) or
                                          'DCNv3' in str(type(module))):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return offsets
    
    @staticmethod
    def visualize_offset_field(offset, sample_stride=8):
        """
        Visualize offset field as vector field
        Args:
            offset: (2*K, H, W) tensor of offsets
            sample_stride: stride for sampling vectors
        """
        # Take first sample point (center of kernel)
        offset_x = offset[0].cpu().numpy()  # X offsets
        offset_y = offset[1].cpu().numpy()  # Y offsets
        
        H, W = offset_x.shape
        
        # Create grid
        y_grid = np.arange(0, H, sample_stride)
        x_grid = np.arange(0, W, sample_stride)
        
        # Sample offsets
        Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')
        
        U = offset_x[Y, X]
        V = offset_y[Y, X]
        
        return X, Y, U, V


# ============================================================================
# Attention Map Generator
# ============================================================================

class AttentionMapGenerator:
    """Generate attention maps from features"""
    
    @staticmethod
    def generate_cam(features, method='mean'):
        """
        Generate Class Activation Map (CAM)
        Args:
            features: (B, C, H, W) feature tensor
            method: 'mean', 'max', or 'norm'
        """
        if method == 'mean':
            cam = features.mean(dim=1, keepdim=True)
        elif method == 'max':
            cam = features.max(dim=1, keepdim=True)[0]
        elif method == 'norm':
            cam = torch.norm(features, dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return cam
    
    @staticmethod
    def normalize_cam(cam):
        """Normalize CAM to [0, 1]"""
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()


# ============================================================================
# Comprehensive Visualization
# ============================================================================

class ModelVisualizer:
    """Main visualization class"""
    
    def __init__(self, model, model_name, device='cuda'):
        self.model = model.to(device).eval()
        self.model_name = model_name
        self.device = device
        self.feature_extractor = FeatureExtractor()
    
    def visualize_sample(self, image, mask, save_path):
        """
        Create comprehensive visualization for one sample
        Args:
            image: input image tensor (1, 3, H, W)
            mask: ground truth mask (1, 1, H, W)
            save_path: path to save visualization
        """
        image = image.to(self.device)
        mask = mask.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            pred = self.model(image)
            pred_prob = torch.sigmoid(pred)
        
        # Convert to numpy for visualization
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        mask_np = mask[0, 0].cpu().numpy()
        pred_np = pred_prob[0, 0].cpu().numpy()
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Basic results
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_np)
        ax1.set_title('Input Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask_np, cmap='gray')
        ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred_np, cmap='gray')
        ax3.set_title('Prediction', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(img_np)
        ax4.imshow(pred_np, cmap='jet', alpha=0.5)
        ax4.set_title('Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[0, 4])
        error_map = np.abs(mask_np - pred_np)
        im = ax5.imshow(error_map, cmap='hot')
        ax5.set_title('Error Map', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, fraction=0.046)
        
        # Row 2: Feature maps at different scales
        self._visualize_feature_maps(fig, gs, image, row=1)
        
        # Row 3: DCN offsets (if model has DCN)
        if 'DCN' in self.model_name:
            self._visualize_dcn_offsets(fig, gs, image, row=2)
        else:
            # Show attention maps instead
            self._visualize_attention_maps(fig, gs, image, row=2)
        
        # Add title
        fig.suptitle(f'{self.model_name} - Visualization', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to: {save_path}")
    
    def _visualize_feature_maps(self, fig, gs, image, row):
        """Visualize feature maps from different decoder levels"""
        
        # Register hooks for encoder/decoder features
        # First, let's detect the model structure dynamically
        layer_names = self._get_layer_paths()
        
        if not layer_names:
            # If no layers found, show message
            ax = fig.add_subplot(gs[row, :])
            ax.text(0.5, 0.5, 'Feature extraction not available for this model', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        self.feature_extractor.register_hooks(self.model, layer_names)
        
        # Forward pass to extract features
        with torch.no_grad():
            _ = self.model(image)
        
        features = self.feature_extractor.features
        
        # Visualize selected features
        feature_keys = list(features.keys())[:5]
        
        for i, key in enumerate(feature_keys):
            if i >= 5:
                break
            
            if key in features:
                ax = fig.add_subplot(gs[row, i])
                
                # Generate attention map
                feat = features[key]
                cam = AttentionMapGenerator.generate_cam(feat, method='norm')
                cam_np = AttentionMapGenerator.normalize_cam(cam)
                
                im = ax.imshow(cam_np, cmap='jet')
                ax.set_title(f'{key}\n({feat.shape[1]}ch, {feat.shape[2]}x{feat.shape[3]})', 
                           fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        self.feature_extractor.clear()
        self.feature_extractor.remove_hooks()
    
    def _get_layer_paths(self):
        """
        Dynamically find important layers to visualize
        Returns dict of layer_name -> module_path
        """
        layer_paths = {}
        
        # Collect all conv layers with reasonable sizes
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 16 <= module.out_channels <= 512:
                    conv_layers.append((name, module.out_channels))
        
        if not conv_layers:
            return layer_paths
        
        # Separate encoder and decoder layers
        encoder_layers = [(n, c) for n, c in conv_layers if 'encoder' in n.lower()]
        decoder_layers = [(n, c) for n, c in conv_layers if 'decoder' in n.lower()]
        
        # Select layers at different depths
        # Try to get 2-3 encoder layers and 2 decoder layers
        if encoder_layers:
            # Get layers from different depths
            step = max(1, len(encoder_layers) // 3)
            selected_encoder = encoder_layers[::step][:3]
            
            for i, (name, channels) in enumerate(selected_encoder):
                layer_paths[f'encoder_{i}'] = name
        
        if decoder_layers:
            # Get first and last decoder layers
            if len(decoder_layers) >= 2:
                layer_paths['decoder_early'] = decoder_layers[0][0]
                layer_paths['decoder_late'] = decoder_layers[-1][0]
            elif len(decoder_layers) == 1:
                layer_paths['decoder'] = decoder_layers[0][0]
        
        return layer_paths
    
    def _visualize_dcn_offsets(self, fig, gs, image, row):
        """Visualize DCN offset fields"""
        
        # Extract offsets
        offsets = DCNOffsetVisualizer.extract_offsets(self.model, image)
        
        if not offsets:
            # No DCN modules found
            ax = fig.add_subplot(gs[row, :])
            ax.text(0.5, 0.5, 'No DCN offsets to visualize', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Visualize first 5 offset fields
        offset_items = list(offsets.items())[:5]
        
        for i, (name, offset) in enumerate(offset_items):
            if i >= 5:
                break
            
            ax = fig.add_subplot(gs[row, i])
            
            # Get offset field
            offset_data = offset[0]  # First sample in batch
            X, Y, U, V = DCNOffsetVisualizer.visualize_offset_field(
                offset_data, sample_stride=8
            )
            
            # Compute offset magnitude for background
            magnitude = np.sqrt(U**2 + V**2)
            
            # Plot
            im = ax.imshow(magnitude, cmap='viridis', alpha=0.6)
            ax.quiver(X, Y, U, V, color='red', alpha=0.8, scale=50)
            
            layer_name = name.split('.')[-2] if '.' in name else name
            ax.set_title(f'DCN Offsets\n{layer_name}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    def _visualize_attention_maps(self, fig, gs, image, row):
        """Visualize attention maps for baseline models"""
        
        # For baseline models, show gradient-based attention
        # This requires grad-cam or similar techniques
        
        ax = fig.add_subplot(gs[row, :])
        ax.text(0.5, 0.5, 
               'Baseline model - No DCN offsets\n(Feature maps shown in row above)', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')


# ============================================================================
# Comparison Visualization
# ============================================================================

class ComparisonVisualizer:
    """Compare two models side by side"""
    
    def __init__(self, model1, model1_name, model2, model2_name, device='cuda'):
        self.vis1 = ModelVisualizer(model1, model1_name, device)
        self.vis2 = ModelVisualizer(model2, model2_name, device)
        self.device = device
    
    def compare_sample(self, image, mask, save_path):
        """
        Create side-by-side comparison
        """
        image = image.to(self.device)
        mask = mask.to(self.device)
        
        # Get predictions from both models
        with torch.no_grad():
            pred1 = torch.sigmoid(self.vis1.model(image))
            pred2 = torch.sigmoid(self.vis2.model(image))
        
        # Convert to numpy
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        mask_np = mask[0, 0].cpu().numpy()
        pred1_np = pred1[0, 0].cpu().numpy()
        pred2_np = pred2[0, 0].cpu().numpy()
        
        # Create comparison figure
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        # Row 1: Model 1
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Input Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred1_np, cmap='gray')
        axes[0, 2].set_title(f'{self.vis1.model_name}\nPrediction', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(img_np)
        axes[0, 3].imshow(pred1_np, cmap='jet', alpha=0.5)
        axes[0, 3].set_title('Overlay', fontweight='bold')
        axes[0, 3].axis('off')
        
        error1 = np.abs(mask_np - pred1_np)
        im1 = axes[0, 4].imshow(error1, cmap='hot')
        axes[0, 4].set_title(f'Error (MAE={error1.mean():.4f})', fontweight='bold')
        axes[0, 4].axis('off')
        plt.colorbar(im1, ax=axes[0, 4], fraction=0.046)
        
        # Row 2: Model 2
        axes[1, 0].imshow(img_np)
        axes[1, 0].set_title('Input Image', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask_np, cmap='gray')
        axes[1, 1].set_title('Ground Truth', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(pred2_np, cmap='gray')
        axes[1, 2].set_title(f'{self.vis2.model_name}\nPrediction', fontweight='bold')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(img_np)
        axes[1, 3].imshow(pred2_np, cmap='jet', alpha=0.5)
        axes[1, 3].set_title('Overlay', fontweight='bold')
        axes[1, 3].axis('off')
        
        error2 = np.abs(mask_np - pred2_np)
        im2 = axes[1, 4].imshow(error2, cmap='hot')
        axes[1, 4].set_title(f'Error (MAE={error2.mean():.4f})', fontweight='bold')
        axes[1, 4].axis('off')
        plt.colorbar(im2, ax=axes[1, 4], fraction=0.046)
        
        # Row 3: Difference analysis
        axes[2, 0].axis('off')
        
        diff = pred1_np - pred2_np
        im3 = axes[2, 1].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2, 1].set_title('Prediction Difference\n(Model1 - Model2)', fontweight='bold')
        axes[2, 1].axis('off')
        plt.colorbar(im3, ax=axes[2, 1], fraction=0.046)
        
        # Both predictions
        axes[2, 2].imshow(pred1_np, cmap='Reds', alpha=0.5)
        axes[2, 2].imshow(pred2_np, cmap='Blues', alpha=0.5)
        axes[2, 2].set_title('Both Predictions\n(Red=M1, Blue=M2)', fontweight='bold')
        axes[2, 2].axis('off')
        
        # Agreement map
        agreement = 1 - np.abs(pred1_np - pred2_np)
        im4 = axes[2, 3].imshow(agreement, cmap='YlGn')
        axes[2, 3].set_title(f'Agreement Map\n(Mean={agreement.mean():.4f})', fontweight='bold')
        axes[2, 3].axis('off')
        plt.colorbar(im4, ax=axes[2, 3], fraction=0.046)
        
        # Metrics comparison
        axes[2, 4].axis('off')
        metrics_text = f"""
        Model 1: {self.vis1.model_name}
        MAE: {error1.mean():.4f}
        
        Model 2: {self.vis2.model_name}
        MAE: {error2.mean():.4f}
        
        Difference: {abs(error1.mean() - error2.mean()):.4f}
        Better: {'Model 1' if error1.mean() < error2.mean() else 'Model 2'}
        """
        axes[2, 4].text(0.1, 0.5, metrics_text, fontsize=10, 
                       verticalalignment='center', family='monospace')
        
        plt.suptitle(f'Model Comparison: {self.vis1.model_name} vs {self.vis2.model_name}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison to: {save_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main visualization pipeline"""
    
    print("\n" + "="*80)
    print("ATTENTION MAP & FEATURE VISUALIZATION")
    print("="*80 + "\n")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = "../MHCD_seg"
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    
    # UNet++ models
    unetpp_baseline = UNetPP_B3(n_classes=1)
    unetpp_dcn = UNetPP_DCNv3_COD(n_classes=1)
    
    # UNet3+ models
    unet3plus_baseline = UNet3Plus_B3(n_classes=1)
    unet3plus_dcn = UNet3Plus_DCNv3_COD(n_classes=1)
    
    # Load weights (if available)
    #torch.load(ckpt_path, map_location=device)
    ckpt = torch.load('logs/UNetPP_B3/best_s_measure.pth', map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        unetpp_baseline.load_state_dict(ckpt["model"])
        print(f"  ✓ Loaded best S-measure")
    # Case 2: Checkpoint is raw state_dict
    elif isinstance(ckpt, dict):
        unetpp_baseline.load_state_dict(ckpt)
        print(f"  ✓ Loaded state_dict")
        
    #unetpp_baseline.load_state_dict(torch.load('logs/UNetPP_B3/best_s_measure.pth', map_location=device)["model"])
    unetpp_dcn.load_state_dict(torch.load('logs/UNetPP_DCNv3_COD_EfficientNetB3/best_s_measure.pth', map_location=device))
    unet3plus_baseline.load_state_dict(torch.load('logs/UNet3Plus_B3_20251215_182637/best_s_measure.pth', map_location=device))
    unet3plus_dcn.load_state_dict(torch.load('logs/UNet3Plus_DCNv3_COD_20251215_182651/best_s_measure.pth', map_location=device))
    
    print("✓ Models loaded\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = MHCDDataset(root, "val", img_size=352)
    print(f"✓ Loaded {len(dataset)} validation samples\n")
    
    # Select samples to visualize
    num_samples = 5
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    print(f"Visualizing {num_samples} samples...\n")
    
    for idx in indices:
        image, mask = dataset[idx]
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        print(f"Processing sample {idx}...")
        
        # Individual visualizations
        vis1 = ModelVisualizer(unetpp_baseline, "UNet++_B3", device)
        vis1.visualize_sample(image, mask, 
                             f"{save_dir}/sample_{idx}_unetpp_baseline.png")
        
        vis2 = ModelVisualizer(unetpp_dcn, "UNet++_DCNv2", device)
        vis2.visualize_sample(image, mask,
                             f"{save_dir}/sample_{idx}_unetpp_dcn.png")
        
        vis3 = ModelVisualizer(unet3plus_baseline, "UNet3+_B3", device)
        vis3.visualize_sample(image, mask,
                             f"{save_dir}/sample_{idx}_unet3plus_baseline.png")
        
        vis4 = ModelVisualizer(unet3plus_dcn, "UNet3+_DCNv2", device)
        vis4.visualize_sample(image, mask,
                             f"{save_dir}/sample_{idx}_unet3plus_dcn.png")
        
        # Comparisons
        comp1 = ComparisonVisualizer(unetpp_baseline, "UNet++_B3",
                                     unet3plus_baseline, "UNet3+_B3", device)
        comp1.compare_sample(image, mask,
                            f"{save_dir}/sample_{idx}_baseline_comparison.png")
        
        comp2 = ComparisonVisualizer(unetpp_dcn, "UNet++_DCNv2",
                                     unet3plus_dcn, "UNet3+_DCNv2", device)
        comp2.compare_sample(image, mask,
                            f"{save_dir}/sample_{idx}_dcn_comparison.png")
        
        print(f"  ✓ Sample {idx} complete\n")
    
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {save_dir}/")
    print("\nGenerated files:")
    print("  - Individual model visualizations (feature maps, offsets)")
    print("  - Side-by-side comparisons (baseline vs DCN)")
    print("  - Architecture comparisons (UNet++ vs UNet3+)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
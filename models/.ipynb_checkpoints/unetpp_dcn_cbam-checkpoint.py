"""
UNet++ with DCN + CBAM + BEM for Camouflaged Object Detection
Combines:
- Deformable Convolution (DCN) for adaptive receptive fields
- CBAM for channel and spatial attention
- Boundary Enhancement Module (BEM) for weak edge detection
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from models.deformable_conv import DCNv1Module, DCNv2Module, DCNv3Module, DCNv4Module
from models.cbam import CBAM, CBAMBlock
from models.boundary_enhancement import BoundaryEnhancementModule


# ============================================================================
# UNet++ + DCN + CBAM + BEM
# ============================================================================

class UNetPP_DCN_CBAM_BEM(nn.Module):
    """
    Enhanced UNet++ with DCN, CBAM, and BEM
    
    Architecture flow:
    Input → Encoder → CBAM → DCN → Decoder → BEM → Output
    
    Args:
        n_classes: number of output classes
        encoder: encoder name (e.g., "efficientnet-b3")
        dcn_version: which DCN version to use ("v1", "v2", "v3", "v4")
        use_cbam: whether to use CBAM attention
        cbam_reduction: reduction ratio for CBAM channel attention
        predict_boundary: whether to predict boundary map for boundary loss
    """
    def __init__(
        self, 
        n_classes=1, 
        encoder="efficientnet-b3",
        dcn_version="v2",
        use_cbam=True,
        cbam_reduction=16,
        predict_boundary=True
    ):
        super().__init__()
        
        # Base UNet++ backbone
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        # Get encoder channels dynamically
        encoder_channels = self.backbone.encoder.out_channels
        
        self.use_cbam = use_cbam
        self.dcn_version = dcn_version
        self.predict_boundary = predict_boundary
        
        # ============================================================
        # CBAM Attention Modules (after encoder, before DCN)
        # Applied to high-level features for "what" and "where" to focus
        # ============================================================
        if use_cbam:
            self.cbam_modules = nn.ModuleList([
                CBAMBlock(encoder_channels[3], reduction_ratio=cbam_reduction),
                CBAMBlock(encoder_channels[4], reduction_ratio=cbam_reduction),
                CBAMBlock(encoder_channels[5], reduction_ratio=cbam_reduction),
            ])
        
        # ============================================================
        # DCN Modules (after CBAM, for adaptive feature extraction)
        # ============================================================
        DCN_class = {
            "v1": DCNv1Module,
            "v2": DCNv2Module,
            "v3": DCNv3Module,
            "v4": DCNv4Module,
        }[dcn_version]
        
        if dcn_version == "v1":
            # DCNv1: Apply to last 2 levels
            self.dcn_refiners = nn.ModuleList([
                DCN_class(encoder_channels[4], encoder_channels[4]),
                DCN_class(encoder_channels[5], encoder_channels[5]),
            ])
            self.dcn_levels = [4, 5]
            
        elif dcn_version in ["v2", "v3"]:
            # DCNv2/v3: Apply to last 3 levels
            self.dcn_refiners = nn.ModuleList([
                DCN_class(encoder_channels[3], encoder_channels[3]),
                DCN_class(encoder_channels[4], encoder_channels[4]),
                DCN_class(encoder_channels[5], encoder_channels[5]),
            ])
            self.dcn_levels = [3, 4, 5]
            
        elif dcn_version == "v4":
            # DCNv4: Apply to last 3 levels with group settings
            self.dcn_refiners = nn.ModuleList([
                DCN_class(encoder_channels[3], encoder_channels[3], num_groups=4),
                DCN_class(encoder_channels[4], encoder_channels[4], num_groups=8),
                DCN_class(encoder_channels[5], encoder_channels[5], num_groups=8),
            ])
            self.dcn_levels = [3, 4, 5]
        
        # ============================================================
        # Boundary Enhancement Module (after decoder)
        # ============================================================
        self.bem = BoundaryEnhancementModule(16, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        """
        Forward pass
        Args:
            x: input image (B, 3, H, W)
            return_boundary: whether to return boundary prediction
        Returns:
            If return_boundary=False: mask only
            If return_boundary=True: (mask, boundary_pred)
        """
        # ==================== Encoder ====================
        features = self.backbone.encoder(x)
        
        features_refined = list(features)
        
        # ==================== CBAM Attention ====================
        if self.use_cbam:
            features_refined[3] = self.cbam_modules[0](features_refined[3])
            features_refined[4] = self.cbam_modules[1](features_refined[4])
            features_refined[5] = self.cbam_modules[2](features_refined[5])
        
        # ==================== DCN Refinement ====================
        for i, level in enumerate(self.dcn_levels):
            features_refined[level] = self.dcn_refiners[i](features_refined[level])
        
        # ==================== Decoder ====================
        decoder_output = self.backbone.decoder(features_refined)
        
        # ==================== Boundary Enhancement ====================
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(decoder_output, return_boundary=True)
        else:
            decoder_output = self.bem(decoder_output, return_boundary=False)
        
        # ==================== Segmentation Head ====================
        mask = self.backbone.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask


# ============================================================================
# Specific Model Variants
# ============================================================================

class UNetPP_DCNv1_CBAM_BEM(UNetPP_DCN_CBAM_BEM):
    """UNet++ with DCNv1 + CBAM + BEM"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__(
            n_classes=n_classes,
            encoder=encoder,
            dcn_version="v1",
            use_cbam=True,
            cbam_reduction=16,
            predict_boundary=predict_boundary
        )


class UNetPP_DCNv2_CBAM_BEM(UNetPP_DCN_CBAM_BEM):
    """UNet++ with DCNv2 + CBAM + BEM"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__(
            n_classes=n_classes,
            encoder=encoder,
            dcn_version="v2",
            use_cbam=True,
            cbam_reduction=16,
            predict_boundary=predict_boundary
        )


class UNetPP_DCNv3_CBAM_BEM(UNetPP_DCN_CBAM_BEM):
    """UNet++ with DCNv3 + CBAM + BEM"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__(
            n_classes=n_classes,
            encoder=encoder,
            dcn_version="v3",
            use_cbam=True,
            cbam_reduction=16,
            predict_boundary=predict_boundary
        )


class UNetPP_DCNv4_CBAM_BEM(UNetPP_DCN_CBAM_BEM):
    """UNet++ with DCNv4 + CBAM + BEM"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__(
            n_classes=n_classes,
            encoder=encoder,
            dcn_version="v4",
            use_cbam=True,
            cbam_reduction=16,
            predict_boundary=predict_boundary
        )


# ============================================================================
# Ablation Study Models
# ============================================================================

class UNetPP_BEM(nn.Module):
    """
    UNet++ with BEM (no DCN, no CBAM) - for ablation study
    Tests the contribution of attention + boundary enhancement without DCN and CBAM
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        # Get encoder channels dynamically
        encoder_channels = self.backbone.encoder.out_channels
        
        self.predict_boundary = predict_boundary        
       
        # Boundary enhancement
        self.bem = BoundaryEnhancementModule(16, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        # Encoder
        features = self.backbone.encoder(x)

        # Decoder
        decoder_output = self.backbone.decoder(features)
        
        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(decoder_output, return_boundary=True)
        else:
            decoder_output = self.bem(decoder_output, return_boundary=False)
        
        # Segmentation head
        mask = self.backbone.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask
            
class UNetPP_CBAM_BEM(nn.Module):
    """
    UNet++ with CBAM + BEM (no DCN) - for ablation study
    Tests the contribution of attention + boundary enhancement without DCN
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3", cbam_reduction=16, predict_boundary=True):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        # Get encoder channels dynamically
        encoder_channels = self.backbone.encoder.out_channels
        
        self.predict_boundary = predict_boundary
        
        # CBAM modules for high-level features
        self.cbam_modules = nn.ModuleList([
            CBAMBlock(encoder_channels[1], reduction_ratio=cbam_reduction),
            CBAMBlock(encoder_channels[2], reduction_ratio=cbam_reduction),
            CBAMBlock(encoder_channels[3], reduction_ratio=cbam_reduction),
            CBAMBlock(encoder_channels[4], reduction_ratio=cbam_reduction),
            CBAMBlock(encoder_channels[5], reduction_ratio=cbam_reduction),
        ])
        
        # Boundary enhancement
        self.bem = BoundaryEnhancementModule(16, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        # Encoder
        features = self.backbone.encoder(x)
        
        # Apply CBAM attention
        features_refined = list(features)
        #features_refined[1] = self.cbam_modules[0](features_refined[1])
        #features_refined[2] = self.cbam_modules[1](features_refined[2])
        #features_refined[3] = self.cbam_modules[2](features_refined[3])
        features_refined[4] = self.cbam_modules[3](features_refined[4])
        features_refined[5] = self.cbam_modules[4](features_refined[5])
        
        # Decoder
        decoder_output = self.backbone.decoder(features_refined)
        
        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(decoder_output, return_boundary=True)
        else:
            decoder_output = self.bem(decoder_output, return_boundary=False)
        
        # Segmentation head
        mask = self.backbone.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask


class UNetPP_CBAM(nn.Module):
    """UNet++ with only CBAM (no DCN, no BEM) - for ablation study"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        encoder_channels = self.backbone.encoder.out_channels
        
        self.cbam_modules = nn.ModuleList([
            CBAMBlock(encoder_channels[3], reduction_ratio=16),
            CBAMBlock(encoder_channels[4], reduction_ratio=16),
            CBAMBlock(encoder_channels[5], reduction_ratio=16),
        ])
        
    def forward(self, x):
        features = self.backbone.encoder(x)
        
        features_refined = list(features)
        features_refined[3] = self.cbam_modules[0](features_refined[3])
        features_refined[4] = self.cbam_modules[1](features_refined[4])
        features_refined[5] = self.cbam_modules[2](features_refined[5])
        
        decoder_output = self.backbone.decoder(features_refined)
        mask = self.backbone.segmentation_head(decoder_output)
        
        return mask


class UNetPP_DCN_BEM(nn.Module):
    """UNet++ with DCN + BEM (no CBAM) - for ablation study"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3", dcn_version="v2", predict_boundary=True):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        encoder_channels = self.backbone.encoder.out_channels
        self.predict_boundary = predict_boundary
        
        DCN_class = {
            "v1": DCNv1Module,
            "v2": DCNv2Module,
            "v3": DCNv3Module,
            "v4": DCNv4Module,
        }[dcn_version]
        
        self.dcn_refiners = nn.ModuleList([
            DCN_class(encoder_channels[3], encoder_channels[3]),
            DCN_class(encoder_channels[4], encoder_channels[4]),
            DCN_class(encoder_channels[5], encoder_channels[5]),
        ])
        
        self.bem = BoundaryEnhancementModule(16, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        features = self.backbone.encoder(x)
        
        features_refined = list(features)
        features_refined[3] = self.dcn_refiners[0](features_refined[3])
        features_refined[4] = self.dcn_refiners[1](features_refined[4])
        features_refined[5] = self.dcn_refiners[2](features_refined[5])
        
        decoder_output = self.backbone.decoder(features_refined)
        
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(decoder_output, return_boundary=True)
        else:
            decoder_output = self.bem(decoder_output, return_boundary=False)
        
        mask = self.backbone.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing UNet++ + DCN + CBAM + BEM Models")
    print("="*80)
    
    # Test input
    x = torch.randn(2, 3, 352, 352)
    
    models_to_test = {
        "UNetPP_DCNv1_CBAM_BEM": UNetPP_DCNv1_CBAM_BEM,
        "UNetPP_DCNv2_CBAM_BEM": UNetPP_DCNv2_CBAM_BEM,
        "UNetPP_DCNv3_CBAM_BEM": UNetPP_DCNv3_CBAM_BEM,
        "UNetPP_DCNv4_CBAM_BEM": UNetPP_DCNv4_CBAM_BEM,
        "UNetPP_CBAM": UNetPP_CBAM,
        "UNetPP_BEM": UNetPP_BEM,
        "UNetPP_CBAM_BEM": UNetPP_CBAM_BEM,
        "UNetPP_DCN_BEM": UNetPP_DCN_BEM,
    }
    
    for name, model_class in models.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        model = model_class(n_classes=1)
        model.eval()
        
        with torch.no_grad():
            out = model(x)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Input shape : {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Parameters  : {total_params:,}")
        print(f"Status      : ✓ Success")
    
    print("\n" + "="*80)
    print("All models tested successfully!")
    print("="*80)
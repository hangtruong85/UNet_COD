import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import segmentation_models_pytorch as smp

from .boundary_enhancement import BoundaryEnhancementModule
from .deformable_conv import DCNv1Module,DCNv2Module, DCNv3Module,  DCNv4Module


# ============================================================================
# Main Models - 4 DCN Versions for Comparison
# ============================================================================

class UNetPP_DCNv1_COD(nn.Module):
    """UNet++ with DCNv1 for Camouflaged Object Detection"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        # Get actual encoder channels dynamically
        encoder_channels = self.backbone.encoder.out_channels
        # encoder_channels is like: [3, 24, 32, 48, 136, 384] for B3
        
        # DCNv1 modules at high-level features only (last 2 levels)
        self.dcn_refiners = nn.ModuleList([
            DCNv1Module(encoder_channels[4], encoder_channels[4]),  
            DCNv1Module(encoder_channels[5], encoder_channels[5]),  
        ])
        
        # Boundary enhancement for weak edges
        self.bem = BoundaryEnhancementModule(16)
        
    def forward(self, x):
        # Encoder
        features = self.backbone.encoder(x)
        
        # Apply DCN refinement to high-level features
        features_refined = list(features)
        features_refined[4] = self.dcn_refiners[0](features[4])
        features_refined[5] = self.dcn_refiners[1](features[5])
        
        # Decoder - pass as list
        decoder_output = self.backbone.decoder(features_refined)
        
        # Boundary enhancement
        decoder_output = self.bem(decoder_output)
        
        # Segmentation head
        mask = self.backbone.segmentation_head(decoder_output)
        
        return mask


class UNetPP_DCNv2_COD(nn.Module):
    """UNet++ with DCNv2 for Camouflaged Object Detection"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        # Get actual encoder channels dynamically
        encoder_channels = self.backbone.encoder.out_channels
        
        # DCNv2 with modulation - apply to last 3 levels
        self.dcn_refiners = nn.ModuleList([
            DCNv2Module(encoder_channels[3], encoder_channels[3]),
            DCNv2Module(encoder_channels[4], encoder_channels[4]),
            DCNv2Module(encoder_channels[5], encoder_channels[5]),
        ])
        
        self.bem = BoundaryEnhancementModule(16)
        
    def forward(self, x):
        features = self.backbone.encoder(x)
        
        # More aggressive refinement with DCNv2
        features_refined = list(features)
        features_refined[3] = self.dcn_refiners[0](features[3])
        features_refined[4] = self.dcn_refiners[1](features[4])
        features_refined[5] = self.dcn_refiners[2](features[5])
        
        decoder_output = self.backbone.decoder(features_refined)
        decoder_output = self.bem(decoder_output)
        mask = self.backbone.segmentation_head(decoder_output)
        
        return mask


class UNetPP_DCNv3_COD(nn.Module):
    """UNet++ with DCNv3 for Camouflaged Object Detection"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        encoder_channels = self.backbone.encoder.out_channels
        
        # DCNv3 với group-wise learning
        self.dcn_refiners = nn.ModuleList([
            DCNv3Module(encoder_channels[3], encoder_channels[3], num_groups=4),
            DCNv3Module(encoder_channels[4], encoder_channels[4], num_groups=4),
            DCNv3Module(encoder_channels[5], encoder_channels[5], num_groups=8),
        ])
        
        self.bem = BoundaryEnhancementModule(16)
        
    def forward(self, x):
        features = self.backbone.encoder(x)
        
        features_refined = list(features)
        features_refined[3] = self.dcn_refiners[0](features[3])
        features_refined[4] = self.dcn_refiners[1](features[4])
        features_refined[5] = self.dcn_refiners[2](features[5])
        
        decoder_output = self.backbone.decoder(features_refined)
        decoder_output = self.bem(decoder_output)
        mask = self.backbone.segmentation_head(decoder_output)
        
        return mask


class UNetPP_DCNv4_COD(nn.Module):
    """UNet++ with DCNv4 for Camouflaged Object Detection"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        encoder_channels = self.backbone.encoder.out_channels
        
        # DCNv4 với multi-scale
        self.dcn_refiners = nn.ModuleList([
            DCNv4Module(encoder_channels[3], encoder_channels[3], num_groups=4),
            DCNv4Module(encoder_channels[4], encoder_channels[4], num_groups=8),
            DCNv4Module(encoder_channels[5], encoder_channels[5], num_groups=8),
        ])
        
        self.bem = BoundaryEnhancementModule(16)
        
    def forward(self, x):
        features = self.backbone.encoder(x)
        
        features_refined = list(features)
        features_refined[3] = self.dcn_refiners[0](features[3])
        features_refined[4] = self.dcn_refiners[1](features[4])
        features_refined[5] = self.dcn_refiners[2](features[5])
        
        decoder_output = self.backbone.decoder(features_refined)
        decoder_output = self.bem(decoder_output)
        mask = self.backbone.segmentation_head(decoder_output)
        
        return mask


# ============================================================================
# Comparison Framework
# ============================================================================
class DCN_Comparison_Suite:
    """
    Suite để so sánh 4 versions DCN trên COD dataset
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        self.models = {
            'baseline': UNetPP_B3(n_classes),
            'dcnv1': UNetPP_DCNv1_COD(n_classes, encoder),
            'dcnv2': UNetPP_DCNv2_COD(n_classes, encoder),
            'dcnv3': UNetPP_DCNv3_COD(n_classes, encoder),
            'dcnv4': UNetPP_DCNv4_COD(n_classes, encoder),
        }
        
    def get_model(self, version):
        return self.models[version]
    
    def compare_params(self):
        """So sánh số lượng parameters"""
        for name, model in self.models.items():
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name:10s}: Total={total:,} | Trainable={trainable:,}")
    
    def compare_flops(self, input_size=(1, 3, 352, 352)):
        """So sánh FLOPs (cần thop library)"""
        try:
            from thop import profile
            x = torch.randn(input_size)
            
            for name, model in self.models.items():
                model.eval()
                flops, params = profile(model, inputs=(x,), verbose=False)
                print(f"{name:10s}: FLOPs={flops/1e9:.2f}G | Params={params/1e6:.2f}M")
        except ImportError:
            print("Install thop: pip install thop")

# Helper class từ code gốc
class UNetPP(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b3",
            #encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
    def forward(self, x):
        return self.backbone(x)

# Helper class từ code gốc
class UNetPP_B3(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
    def forward(self, x):
        return self.backbone(x)

# Helper class từ code gốc
class UNetPP_Resnet50(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
    def forward(self, x):
        return self.backbone(x)
        
# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("DCN Comparison Suite for Camouflaged Object Detection")
    print("="*80)
    
    # Initialize comparison suite
    suite = DCN_Comparison_Suite(n_classes=1, encoder="efficientnet-b3")
    
    # Compare parameters
    print("\n[1] Parameter Comparison:")
    print("-"*80)
    suite.compare_params()
    
    # Test forward pass
    print("\n[2] Forward Pass Test:")
    print("-"*80)
    x = torch.randn(2, 3, 352, 352)  # COD datasets thường dùng 352x352
    
    for version in ['baseline', 'dcnv1', 'dcnv2', 'dcnv3', 'dcnv4']:
        model = suite.get_model(version)
        model.eval()
        
        with torch.no_grad():
            out = model(x)
        
        print(f"{version:10s}: Input {x.shape} -> Output {out.shape}")
    
    print("\n[3] Recommended Training Strategy:")
    print("-"*80)
    print("""
    Stage 1: Warmup (5 epochs)
        - Freeze encoder (giữ pretrained weights)
        - Train only DCN modules + decoder
        - LR: 1e-4
    
    Stage 2: Fine-tune (15-20 epochs)
        - Unfreeze toàn bộ
        - LR: 1e-5 for encoder, 1e-4 for DCN
        - Cosine annealing
    
    Loss functions cho COD:
        - BCE + IoU Loss
        - Boundary-aware loss (edge loss)
        - Weighted loss (foreground/background imbalance)
    
    Metrics:
        - S-measure (Structure measure)
        - E-measure (Enhanced-alignment measure)  
        - F-measure (weighted F-beta)
        - MAE (Mean Absolute Error)
    """)
    
    print("\n[4] Expected Results on COD10K:")
    print("-"*80)
    print("""
    Baseline UNet++:        S-measure ~0.750
    + DCNv1:               S-measure ~0.765 (+2%)
    + DCNv2:               S-measure ~0.780 (+4%)
    + DCNv3:               S-measure ~0.790 (+5.3%)
    + DCNv4:               S-measure ~0.800 (+6.7%) <- Best
    
    Note: DCNv4 thường tốt nhất nhưng cũng chậm nhất
          DCNv2 là balance tốt giữa speed và accuracy
    """)
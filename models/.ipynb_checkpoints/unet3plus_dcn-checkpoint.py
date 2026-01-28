import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import segmentation_models_pytorch as smp

from .deformable_conv import DCNv1Module, DCNv2Module, DCNv3Module, DCNv4Module

from .boundary_enhancement import BoundaryEnhancementModule

# ============================================================================
# UNet3+ Decoder Block
# ============================================================================

class UNet3PlusDecoderBlock(nn.Module):
    """
    UNet3+ Decoder Block with full-scale skip connections
    Aggregates features from ALL encoder levels
    """
    def __init__(self, in_channels_list, out_channels, use_dcn=False, dcn_module=None):
        """
        Args:
            in_channels_list: list of input channels from each encoder level
            out_channels: output channels after fusion
            use_dcn: whether to use DCN for feature refinement
            dcn_module: which DCN module to use (DCNv1/v2/v3/v4)
        """
        super().__init__()
        
        # Calculate channels per branch
        cat_channels = out_channels * len(in_channels_list)
        
        # Convs to reduce each encoder feature to out_channels
        self.conv_branches = nn.ModuleList()
        for in_ch in in_channels_list:
            if use_dcn and dcn_module is not None:
                # Use DCN for feature extraction
                self.conv_branches.append(
                    nn.Sequential(
                        dcn_module(in_ch, out_channels),
                    )
                )
            else:
                # Standard conv
                self.conv_branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # Fusion after concatenation
        self.fusion = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, target_size):
        """
        Args:
            features: list of feature maps from all encoder levels
            target_size: (H, W) target size for this decoder level
        """
        processed = []
        
        for i, (feat, conv) in enumerate(zip(features, self.conv_branches)):
            # Resize to target size
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            
            # Process with conv/dcn
            feat = conv(feat)
            processed.append(feat)
        
        # Concatenate and fuse
        cat_feat = torch.cat(processed, dim=1)
        out = self.fusion(cat_feat)
        
        return out

class UNet3Plus(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder (Baseline - no DCN)
    Full-scale skip connections for better feature aggregation
    """
    def __init__(self, n_classes=1):
        super().__init__()
        
        # EfficientNet-B3 encoder
        self.encoder = smp.encoders.get_encoder(
            name="efficientnet-b3",
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Encoder output channels: [3, 40, 32, 48, 136, 384]
        encoder_channels = self.encoder.out_channels
        
        # Decoder channels (progressively decrease)
        decoder_channels = 64
        
        # UNet3+ decoder blocks (4 decoder levels)
        # Each block aggregates features from ALL encoder levels
        
        # Decoder 4 (deepest): aggregate from e1, e2, e3, e4, e5
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], encoder_channels[5]],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Decoder 3: aggregate from e1, e2, e3, e4, d4
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], decoder_channels],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Decoder 2: aggregate from e1, e2, e3, d3, d4
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels, decoder_channels],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Decoder 1: aggregate from e1, e2, d2, d3, d4
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]
        
        # Calculate target sizes for each decoder level
        size_d4 = (features[4].shape[2], features[4].shape[3])  # Same as e4
        size_d3 = (features[3].shape[2], features[3].shape[3])  # Same as e3
        size_d2 = (features[2].shape[2], features[2].shape[3])  # Same as e2
        size_d1 = (features[1].shape[2], features[1].shape[3])  # Same as e1
        
        # Decoder 4
        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        
        # Decoder 3
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        
        # Decoder 2
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        
        # Decoder 1
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask

# ============================================================================
# UNet3+ Base Model
# ============================================================================

class UNet3Plus_B3(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder (Baseline - no DCN)
    Full-scale skip connections for better feature aggregation
    """
    def __init__(self, n_classes=1):
        super().__init__()
        
        # EfficientNet-B3 encoder
        self.encoder = smp.encoders.get_encoder(
            name="efficientnet-b3",
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Encoder output channels: [3, 40, 32, 48, 136, 384]
        encoder_channels = self.encoder.out_channels
        
        # Decoder channels (progressively decrease)
        decoder_channels = 64
        
        # UNet3+ decoder blocks (4 decoder levels)
        # Each block aggregates features from ALL encoder levels
        
        # Decoder 4 (deepest): aggregate from e1, e2, e3, e4, e5
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], encoder_channels[5]],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Decoder 3: aggregate from e1, e2, e3, e4, d4
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], decoder_channels],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Decoder 2: aggregate from e1, e2, e3, d3, d4
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels, decoder_channels],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Decoder 1: aggregate from e1, e2, d2, d3, d4
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels,
            use_dcn=False
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]
        
        # Calculate target sizes for each decoder level
        size_d4 = (features[4].shape[2], features[4].shape[3])  # Same as e4
        size_d3 = (features[3].shape[2], features[3].shape[3])  # Same as e3
        size_d2 = (features[2].shape[2], features[2].shape[3])  # Same as e2
        size_d1 = (features[1].shape[2], features[1].shape[3])  # Same as e1
        
        # Decoder 4
        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        
        # Decoder 3
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        
        # Decoder 2
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        
        # Decoder 1
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask


# ============================================================================
# UNet3+ with DCN Models
# ============================================================================

class UNet3Plus_DCNv1_COD(nn.Module):
    """UNet3+ with DCNv1 for COD"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        encoder_channels = self.encoder.out_channels
        decoder_channels = 64
        
        # UNet3+ decoders with DCNv1
        self.decoder4 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], encoder_channels[5]],
            decoder_channels, use_dcn=True, dcn_module=DCNv1Module
        )
        
        self.decoder3 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv1Module
        )
        
        self.decoder2 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv1Module
        )
        
        self.decoder1 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv1Module
        )
        
        self.bem = BoundaryEnhancementModule(decoder_channels)
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        
        size_d4 = (features[4].shape[2], features[4].shape[3])
        size_d3 = (features[3].shape[2], features[3].shape[3])
        size_d2 = (features[2].shape[2], features[2].shape[3])
        size_d1 = (features[1].shape[2], features[1].shape[3])
        
        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)
        
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        d1 = self.bem(d1)
        mask = self.segmentation_head(d1)
        
        return mask


class UNet3Plus_DCNv2_COD(nn.Module):
    """UNet3+ with DCNv2 for COD"""
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        encoder_channels = self.encoder.out_channels
        decoder_channels = 64
        
        self.decoder4 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], encoder_channels[5]],
            decoder_channels, use_dcn=True, dcn_module=DCNv2Module
        )
        
        self.decoder3 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv2Module
        )
        
        self.decoder2 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv2Module
        )
        
        self.decoder1 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv2Module
        )
        
        self.bem = BoundaryEnhancementModule(decoder_channels)
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        
        size_d4 = (features[4].shape[2], features[4].shape[3])
        size_d3 = (features[3].shape[2], features[3].shape[3])
        size_d2 = (features[2].shape[2], features[2].shape[3])
        size_d1 = (features[1].shape[2], features[1].shape[3])
        
        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)
        
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        d1 = self.bem(d1)
        mask = self.segmentation_head(d1)
        
        return mask


class UNet3Plus_DCNv3_COD(nn.Module):
    """UNet3+ with DCNv3 for COD"""
    def __init__(self, n_classes=1):
        super().__init__()
        
        self.encoder = smp.encoders.get_encoder(
            name="efficientnet-b3",
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        encoder_channels = self.encoder.out_channels
        decoder_channels = 64
        
        self.decoder4 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], encoder_channels[5]],
            decoder_channels, use_dcn=True, dcn_module=DCNv3Module
        )
        
        self.decoder3 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv3Module
        )
        
        self.decoder2 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv3Module
        )
        
        self.decoder1 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv3Module
        )
        
        self.bem = BoundaryEnhancementModule(decoder_channels)
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        
        size_d4 = (features[4].shape[2], features[4].shape[3])
        size_d3 = (features[3].shape[2], features[3].shape[3])
        size_d2 = (features[2].shape[2], features[2].shape[3])
        size_d1 = (features[1].shape[2], features[1].shape[3])
        
        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)
        
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        d1 = self.bem(d1)
        mask = self.segmentation_head(d1)
        
        return mask


class UNet3Plus_DCNv4_COD(nn.Module):
    """UNet3+ with DCNv4 for COD"""
    def __init__(self, n_classes=1):
        super().__init__()
        
        self.encoder = smp.encoders.get_encoder(
            name="efficientnet-b3",
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        encoder_channels = self.encoder.out_channels
        decoder_channels = 64
        
        self.decoder4 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], encoder_channels[5]],
            decoder_channels, use_dcn=True, dcn_module=DCNv4Module
        )
        
        self.decoder3 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4], decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv4Module
        )
        
        self.decoder2 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv4Module
        )
        
        self.decoder1 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels, decoder_channels],
            decoder_channels, use_dcn=True, dcn_module=DCNv4Module
        )
        
        self.bem = BoundaryEnhancementModule(decoder_channels)
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        
        size_d4 = (features[4].shape[2], features[4].shape[3])
        size_d3 = (features[3].shape[2], features[3].shape[3])
        size_d2 = (features[2].shape[2], features[2].shape[3])
        size_d1 = (features[1].shape[2], features[1].shape[3])
        
        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)
        
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        d1 = self.bem(d1)
        mask = self.segmentation_head(d1)
        
        return mask


# ============================================================================
# Testing
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("UNet3+ with DCN - Model Testing")
    print("="*80)
    
    models = {
        "UNet3Plus": UNet3Plus,
        "UNet3Plus_B3": UNet3Plus_B3,
        "UNet3Plus_DCNv1_COD": UNet3Plus_DCNv1_COD,
        "UNet3Plus_DCNv2_COD": UNet3Plus_DCNv2_COD,
        "UNet3Plus_DCNv3_COD": UNet3Plus_DCNv3_COD,
        "UNet3Plus_DCNv4_COD": UNet3Plus_DCNv4_COD,
    }
    
    x = torch.randn(2, 3, 352, 352)
    
    for name, model_cls in models.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        model = model_cls(n_classes=1)
        model.eval()
        
        with torch.no_grad():
            out = model(x)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Input shape : {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Parameters  : {total_params:,}")
        print(f"Status      : ✓ Success")
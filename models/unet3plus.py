"""
UNet3+ models for Camouflaged Object Detection
Full-scale skip connections for better multi-scale feature aggregation
Variants: UNet3Plus (baseline), UNet3Plus_B3, UNet3Plus_BEM (with Boundary Enhancement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .boundary_enhancement import BoundaryEnhancementModule


# ============================================================================
# UNet3+ Decoder Block
# ============================================================================

class UNet3PlusDecoderBlock(nn.Module):
    """
    UNet3+ Decoder Block with full-scale skip connections.
    Aggregates features from ALL encoder levels via conv + resize + concat + fuse.
    """
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: list of input channels from each source (encoder/decoder levels)
            out_channels: output channels after fusion
        """
        super().__init__()

        cat_channels = out_channels * len(in_channels_list)

        # Conv branch to reduce each source feature to out_channels
        self.conv_branches = nn.ModuleList()
        for in_ch in in_channels_list:
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
            features: list of feature maps from all source levels
            target_size: (H, W) target spatial size for this decoder level
        """
        processed = []

        for feat, conv in zip(features, self.conv_branches):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            feat = conv(feat)
            processed.append(feat)

        cat_feat = torch.cat(processed, dim=1)
        out = self.fusion(cat_feat)

        return out


# ============================================================================
# UNet3+ Models
# ============================================================================

class UNet3Plus(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder (baseline).
    Full-scale skip connections for better feature aggregation.
    """
    def __init__(self, n_classes=1, encoder='efficientnet-b3'):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )

        # Encoder output channels: e.g. [3, 40, 32, 48, 136, 384] for efficientnet-b3
        encoder_channels = self.encoder.out_channels
        decoder_channels = 64

        # Decoder 4 (deepest): aggregate from e1, e2, e3, e4, e5
        self.decoder4 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3],
             encoder_channels[4], encoder_channels[5]],
            decoder_channels
        )

        # Decoder 3: aggregate from e1, e2, e3, e4, d4
        self.decoder3 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3],
             encoder_channels[4], decoder_channels],
            decoder_channels
        )

        # Decoder 2: aggregate from e1, e2, e3, d3, d4
        self.decoder2 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3],
             decoder_channels, decoder_channels],
            decoder_channels
        )

        # Decoder 1: aggregate from e1, e2, d2, d3, d4
        self.decoder1 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2],
             decoder_channels, decoder_channels, decoder_channels],
            decoder_channels
        )

        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]

        # Target sizes for each decoder level
        size_d4 = features[4].shape[2:]
        size_d3 = features[3].shape[2:]
        size_d2 = features[2].shape[2:]
        size_d1 = features[1].shape[2:]

        # Decode
        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)

        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)

        mask = self.segmentation_head(d1)
        return mask


# Alias for naming consistency
UNet3Plus_B3 = UNet3Plus


class UNet3Plus_BEM(nn.Module):
    """
    UNet3+ + Boundary Enhancement Module.
    Returns (mask, boundary) when predict_boundary=True.
    """
    def __init__(self, n_classes=1, encoder='efficientnet-b3', predict_boundary=True):
        super().__init__()
        self.predict_boundary = predict_boundary

        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )

        encoder_channels = self.encoder.out_channels
        decoder_channels = 64

        self.decoder4 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3],
             encoder_channels[4], encoder_channels[5]],
            decoder_channels
        )
        self.decoder3 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3],
             encoder_channels[4], decoder_channels],
            decoder_channels
        )
        self.decoder2 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2], encoder_channels[3],
             decoder_channels, decoder_channels],
            decoder_channels
        )
        self.decoder1 = UNet3PlusDecoderBlock(
            [encoder_channels[1], encoder_channels[2],
             decoder_channels, decoder_channels, decoder_channels],
            decoder_channels
        )

        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.encoder(x)

        size_d4 = features[4].shape[2:]
        size_d3 = features[3].shape[2:]
        size_d2 = features[2].shape[2:]
        size_d1 = features[1].shape[2:]

        d4 = self.decoder4([features[1], features[2], features[3], features[4], features[5]], size_d4)
        d3 = self.decoder3([features[1], features[2], features[3], features[4], d4], size_d3)
        d2 = self.decoder2([features[1], features[2], features[3], d3, d4], size_d2)
        d1 = self.decoder1([features[1], features[2], d2, d3, d4], size_d1)

        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)

        enhanced = self.bem(d1, return_boundary=self.predict_boundary)

        if self.predict_boundary:
            enhanced_features, boundary = enhanced
            mask = self.segmentation_head(enhanced_features)
            return mask, boundary
        else:
            mask = self.segmentation_head(enhanced)
            return mask


if __name__ == "__main__":
    x = torch.randn(2, 3, 352, 352)

    for name, ModelClass in [("UNet3Plus", UNet3Plus), ("UNet3Plus_BEM", UNet3Plus_BEM)]:
        model = ModelClass(n_classes=1)
        model.eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6

        with torch.no_grad():
            out = model(x)

        if isinstance(out, tuple):
            print(f"{name}: {params:.2f}M params | mask={out[0].shape}, boundary={out[1].shape}")
        else:
            print(f"{name}: {params:.2f}M params | output={out.shape}")

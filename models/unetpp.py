"""
UNet++ models for Camouflaged Object Detection
Variants: UNetPP (baseline), UNetPP_B3 (EfficientNet-B3), UNetPP_BEM (with Boundary Enhancement)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .boundary_enhancement import BoundaryEnhancementModule


class UNetPP(nn.Module):
    """
    UNet++ with timm-efficientnet-b3 encoder
    """
    def __init__(self, n_classes=1, encoder='timm-efficientnet-b3'):
        super().__init__()
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
            decoder_attention_type=None,
        )

    def forward(self, x):
        return self.base(x)


class UNetPP_B3(nn.Module):
    """
    UNet++ with efficientnet-b3 encoder
    """
    def __init__(self, n_classes=1, encoder='efficientnet-b3'):
        super().__init__()
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
            decoder_attention_type=None,
        )

    def forward(self, x):
        return self.base(x)


class UNetPP_BEM(nn.Module):
    """
    UNet++ + Boundary Enhancement Module
    Returns (mask, boundary) when predict_boundary=True
    """
    def __init__(self, n_classes=1, encoder='timm-efficientnet-b3', predict_boundary=True):
        super().__init__()
        self.predict_boundary = predict_boundary
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
            decoder_attention_type=None,
        )
        # Decoder output channels = 16 (last element of decoder_channels)
        self.bem = BoundaryEnhancementModule(16, predict_boundary=predict_boundary)

    def forward(self, x):
        features = self.base.encoder(x)
        decoder_output = self.base.decoder(*features)

        enhanced = self.bem(decoder_output, return_boundary=self.predict_boundary)

        if self.predict_boundary:
            enhanced_features, boundary = enhanced
            masks = self.base.segmentation_head(enhanced_features)
            return masks, boundary
        else:
            masks = self.base.segmentation_head(enhanced)
            return masks


if __name__ == "__main__":
    x = torch.randn(2, 3, 352, 352)

    for name, ModelClass in [("UNetPP", UNetPP), ("UNetPP_B3", UNetPP_B3), ("UNetPP_BEM", UNetPP_BEM)]:
        model = ModelClass(n_classes=1)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        out = model(x)
        if isinstance(out, tuple):
            print(f"{name}: {params:.2f}M params | mask={out[0].shape}, boundary={out[1].shape}")
        else:
            print(f"{name}: {params:.2f}M params | output={out.shape}")

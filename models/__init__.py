"""
Models package for Camouflaged Object Detection

Architectures:
  - UNet:    Standard encoder-decoder with skip connections
  - UNetPP:  UNet++ with nested dense skip connections
  - UNet3Plus: UNet3+ with full-scale skip connections

Each architecture has:
  - Baseline variant (EfficientNet-B3 encoder)
  - BEM variant (+ Boundary Enhancement Module)
"""

from .boundary_enhancement import BoundaryEnhancementModule

# UNet models
from .unet import UNet, UNet_B3, UNet_BEM

# UNet++ models
from .unetpp import UNetPP, UNetPP_B3, UNetPP_BEM

# UNet3+ models
from .unet3plus import UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM

__all__ = [
    'BoundaryEnhancementModule',

    # UNet
    'UNet', 'UNet_B3', 'UNet_BEM',

    # UNet++
    'UNetPP', 'UNetPP_B3', 'UNetPP_BEM',

    # UNet3+
    'UNet3Plus', 'UNet3Plus_B3', 'UNet3Plus_BEM',
]

"""
Models package for Camouflaged Object Detection
"""

# Import modules
from .deformable_conv import DCNv1Module, DCNv2Module, DCNv3Module, DCNv4Module
from .cbam import CBAM, CBAMBlock, ChannelAttention, SpatialAttention
from .boundary_enhancement import BoundaryEnhancementModule

# Import baseline models
from .unetpp_bem import (
    UNetPP,
    UNetPP_B3,
    UNetPP_DCNv1_COD,
    UNetPP_DCNv2_COD,
    UNetPP_DCNv3_COD,
    UNetPP_DCNv4_COD
)

# Import UNet3+ models
from .unet3plus_dcn import (
    UNet3Plus_B3,
    UNet3Plus_DCNv1_COD,
    UNet3Plus_DCNv2_COD,
    UNet3Plus_DCNv3_COD,
    UNet3Plus_DCNv4_COD
)

# Import new models with DCN + CBAM + BEM
from .unetpp_dcn_cbam import (
    UNetPP_DCN_CBAM_BEM,
    UNetPP_DCNv1_CBAM_BEM,
    UNetPP_DCNv2_CBAM_BEM,
    UNetPP_DCNv3_CBAM_BEM,
    UNetPP_DCNv4_CBAM_BEM,
    UNetPP_CBAM,
    UNetPP_DCN_BEM,
    UNetPP_BEM
)

__all__ = [
    # Modules
    'DCNv1Module', 'DCNv2Module', 'DCNv3Module', 'DCNv4Module',
    'CBAM', 'CBAMBlock', 'ChannelAttention', 'SpatialAttention',
    'BoundaryEnhancementModule',
    
    # UNet++ baseline
    'UNetPP', 'UNetPP_B3',
    'UNetPP_DCNv1_COD', 'UNetPP_DCNv2_COD', 
    'UNetPP_DCNv3_COD', 'UNetPP_DCNv4_COD',
    
    # UNet3+
    'UNet3Plus_B3',
    'UNet3Plus_DCNv1_COD', 'UNet3Plus_DCNv2_COD',
    'UNet3Plus_DCNv3_COD', 'UNet3Plus_DCNv4_COD',
    
    # New models
    'UNetPP_DCN_CBAM_BEM',
    'UNetPP_DCNv1_CBAM_BEM', 'UNetPP_DCNv2_CBAM_BEM',
    'UNetPP_DCNv3_CBAM_BEM', 'UNetPP_DCNv4_CBAM_BEM',
    'UNetPP_CBAM', 'UNetPP_DCN_BEM', 'UNetPP_BEM'
]
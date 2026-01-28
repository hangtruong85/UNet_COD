## UNet++ + EfficientNet-B2 + BEM (DeformConv2d)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import segmentation_models_pytorch as smp

class BEM(nn.Module):
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.offset = nn.Conv2d(in_ch, 18, 3, padding=1)
        self.deform = DeformConv2d(in_ch, mid_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(mid_ch)
        self.out = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, feat, target_size):
        off = self.offset(feat)
        x = self.deform(feat, off)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.out(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class UNetPP_B2_BEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            classes=1,
            activation=None,
        )

        ch_last = self.backbone.encoder.out_channels[-1]
        self.bem = BEM(ch_last)

    def forward(self, x):
        # encode ra multi-scale feature maps
        feats = self.backbone.encoder(x)
    
        # BEM refine encoder last feature
        B_feat = self.bem(feats[-1], feats[-1].shape[-2:])
    
        # inject
        feats[-1] = feats[-1] + B_feat
    
        # decode với features đã enhanced
        dec = self.backbone.decoder(feats)
    
        # segmentation head chuẩn SMP → output mask
        mask = self.backbone.segmentation_head(dec)
    
        return mask, B_feat


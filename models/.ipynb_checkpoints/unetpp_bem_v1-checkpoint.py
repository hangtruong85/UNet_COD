import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import segmentation_models_pytorch as smp


class BEM(nn.Module):
    """
    Boundary Extraction Module — output feature not 1-channel mask
    Đây là khác biệt lớn so với bản auxiliary.
    """
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.offset = nn.Conv2d(in_ch, 18, 3, padding=1)
        self.deform = DeformConv2d(in_ch, mid_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat, target_size):
        offset = self.offset(feat)
        x = self.deform(feat, offset)
        x = self.bn(x)
        x = self.relu(x)
        # upsample về size decoder output
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x

class UNetPP_B2_BEM_X00(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            classes=1,
            activation=None
        )

        ch0 = self.backbone.encoder.out_channels[0]
        self.bem = BEM(ch0, mid_ch=64)

        self.boundary_head = nn.Conv2d(64, 1, 1)
        
        self.fuse_BEM = nn.Conv2d(64, ch0, kernel_size=1)


    def forward(self, x):
        feats = self.backbone.encoder(x)
    
        # X_0,0
        X00 = feats[0]
    
        # BEM latent
        B00 = self.bem(X00, X00.shape[-2:])
    
        # align channel
        B00_align = self.fuse_BEM(B00)
    
        # inject BEM into node X_0,0
        feats[0] = X00 + B00_align
    
        # decoder++ tree
        dec = self.backbone.decoder(feats)
        mask = self.backbone.segmentation_head(dec)
    
        # boundary supervision head
        boundary_pred = self.boundary_head(B00)
        boundary_pred = F.interpolate(
            boundary_pred,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        return mask, boundary_pred


class UNetPP_B2_BEM(nn.Module):
    """
    Feature Fusion: boundary feature tham gia trực tiếp vào segmentation
    """
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            classes=1,
            activation=None,
        )

        # lấy channel của encoder cuối
        ch_last = self.backbone.encoder.out_channels[-1]

        # BEM sinh feature boundary (chứ không phải map biên 1 kênh)
        self.bem = BEM(ch_last, mid_ch=64)

        # fusion = concat(decoder_out, bem_feat)
        # decoder_out thường 64/128 channels tùy SMP
        dec_out_ch = self.backbone.decoder.out_channels[-1]

        # segmentation head mới
        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_out_ch + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
        # map from B_feat (64) -> encoder_last (352)
        self.boundary_proj = nn.Conv2d(64, self.backbone.encoder.out_channels[-1], 1)
        
        # boundary map head (latent → 1 channel)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 1) Encoder multi-scale
        feats = self.backbone.encoder(x)
    
        # 2) BEM refine latent space (size encoder cuối)
        B_feat = self.bem(feats[-1], feats[-1].shape[-2:])     # latent: Cb x H/32 x W/32
    
        # 3) Boundary-guided feature injection into encoder last layer
        enc_ref = self.boundary_proj(B_feat)                  # → match channel 352
        g = torch.sigmoid(enc_ref)                            # gating mask
        feats[-1] = feats[-1] * (1 + g)                       # multiplicative refine
    
        # 4) Decode normally with UNet++ decoder
        dec = self.backbone.decoder(feats)
    
        # 5) Segmentation head
        mask = self.backbone.segmentation_head(dec)
    
        # 6) Boundary prediction head (1-channel supervision target)
        boundary_pred = self.boundary_head(B_feat)            # convert latent → 1 ch map
        boundary_pred = F.interpolate(
            boundary_pred,
            size=mask.shape[-2:],                             # match GT resolution
            mode="bilinear",
            align_corners=False,
        )
    
        return mask, boundary_pred


class UNetPP_B3_BEM(nn.Module):
    """
    Feature Fusion: boundary feature tham gia trực tiếp vào segmentation
    """
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            classes=1,
            activation=None,
        )

        # lấy channel của encoder cuối
        ch_last = self.backbone.encoder.out_channels[-1]

        # BEM sinh feature boundary (chứ không phải map biên 1 kênh)
        self.bem = BEM(ch_last, mid_ch=64)

        # fusion = concat(decoder_out, bem_feat)
        # decoder_out thường 64/128 channels tùy SMP
        dec_out_ch = self.backbone.decoder.out_channels[-1]

        # segmentation head mới
        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_out_ch + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
        # map from B_feat (64) -> encoder_last (352)
        self.boundary_proj = nn.Conv2d(64, self.backbone.encoder.out_channels[-1], 1)
        
        # boundary map head (latent → 1 channel)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 1) Encoder multi-scale
        feats = self.backbone.encoder(x)
    
        # 2) BEM refine latent space (size encoder cuối)
        B_feat = self.bem(feats[-1], feats[-1].shape[-2:])     # latent: Cb x H/32 x W/32
    
        # 3) Boundary-guided feature injection into encoder last layer
        enc_ref = self.boundary_proj(B_feat)                  # → match channel 352
        g = torch.sigmoid(enc_ref)                            # gating mask
        feats[-1] = feats[-1] * (1 + g)                       # multiplicative refine
    
        # 4) Decode normally with UNet++ decoder
        dec = self.backbone.decoder(feats)
    
        # 5) Segmentation head
        mask = self.backbone.segmentation_head(dec)
    
        # 6) Boundary prediction head (1-channel supervision target)
        boundary_pred = self.boundary_head(B_feat)            # convert latent → 1 ch map
        boundary_pred = F.interpolate(
            boundary_pred,
            size=mask.shape[-2:],                             # match GT resolution
            mode="bilinear",
            align_corners=False,
        )
    
        return mask, boundary_pred
        
class UNetPP(nn.Module):
    """
    Baseline UNet++ using ResNet encoder (no BEM, no injection)
    """
    def __init__(self, n_classes=1, encoder="resnet34"):
        super().__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        mask = self.backbone(x)
        return mask


class UNetPP_B1(nn.Module):
    """
    Baseline UNet++ using EfficientNet-B1 encoder (no BEM, no injection)
    """
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b1",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        mask = self.backbone(x)
        return mask
        
class UNetPP_B2(nn.Module):
    """
    Baseline UNet++ using EfficientNet-B2 encoder (no BEM, no injection)
    """
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        mask = self.backbone(x)
        return mask

class UNetPP_B3(nn.Module):
    """
    Baseline UNet++ using EfficientNet-B3 encoder (no BEM, no injection)
    """
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        mask = self.backbone(x)
        return mask



class UNetPP_B4(nn.Module):
    """
    Baseline UNet++ using EfficientNet-B4 encoder (no BEM, no injection)
    """
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        mask = self.backbone(x)
        return mask
        
class UNetPP_B5(nn.Module):
    """
    Baseline UNet++ using EfficientNet-B5 encoder (no BEM, no injection)
    """
    def __init__(self, n_classes=1):
        super().__init__()

        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        mask = self.backbone(x)
        return mask
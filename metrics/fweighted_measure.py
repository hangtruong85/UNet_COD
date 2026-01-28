import torch
import torch.nn.functional as F


def gaussian_kernel(channel, kernel_size=7, sigma=5):
    grid = torch.arange(kernel_size).float()
    mean = (kernel_size - 1) / 2
    kernel = torch.exp(-(grid - mean) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    g2d = torch.outer(kernel, kernel)
    g2d = g2d.unsqueeze(0).unsqueeze(0)
    return g2d.repeat(channel, 1, 1, 1)


def fw_measure(pred, gt, beta2=0.3*0.3):
    """
    pred, gt ∈ [0,1], shape (1,H,W)
    Weighted F-measure from CVPR2014

    Steps:
    1) absolute error map
    2) spatial weighting (Gaussian)
    3) weighted precision / recall
    """

    pred = pred.squeeze().float()
    gt   = gt.squeeze().float()

    # step1: error map
    E = torch.abs(pred - gt)

    # step2: Gaussian blur
    g = gaussian_kernel(1, kernel_size=7, sigma=5).to(pred.device)
    E_blur = F.conv2d(E.unsqueeze(0).unsqueeze(0), g, padding=3)[0,0]

    # step3: Weighting
    TP = (gt * (1 - E_blur)).sum()
    FP = ((1 - gt) * (1 - E_blur)).sum()

    FN = (gt * E_blur).sum()

    P_w = TP / (TP + FP + 1e-7)
    R_w = TP / (TP + FN + 1e-7)

    F_w = (1 + beta2) * P_w * R_w / (beta2 * P_w + R_w + 1e-7)
    return F_w

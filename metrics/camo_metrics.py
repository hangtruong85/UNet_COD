import torch
import torch.nn.functional as F


def mae(pred, gt):
    pred = torch.sigmoid(pred)
    return torch.mean(torch.abs(pred - gt))


def s_measure(pred, gt, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.clamp(0,1)
    gt = gt.clamp(0,1)

    fg = pred*gt
    bg = (1-pred)*(1-gt)
    return (fg.mean()+bg.mean())/2


def f_measure(pred, gt, beta2=0.3*0.3, eps=1e-6):
    p = torch.sigmoid(pred) > 0.5
    g = gt > 0.5

    tp = (p*g).sum()
    fp = (p & ~g).sum()
    fn = (~p & g).sum()

    prec = tp/(tp+fp+eps)
    rec  = tp/(tp+fn+eps)
    return (1+beta2)*prec*rec/(beta2*prec+rec+eps)


def e_measure(pred, gt, eps=1e-6):
    p = torch.sigmoid(pred)
    pm = p.mean()
    gm = gt.mean()
    align = (2*(p-pm)*(gt-gm)+eps)/(((p-pm)**2)+((gt-gm)**2)+eps)
    enhance = ((align+1)**2)/4
    return enhance.mean()

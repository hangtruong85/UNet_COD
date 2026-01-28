import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from datasets.mhcd_dataset import MHCDDataset

from models.unetpp_bem import UNetPP, UNetPP_B1, UNetPP_B2, UNetPP_B3, UNetPP_B5, UNetPP_B2_BEM, UNetPP_B3_BEM, UNetPP_B2_BEM_X00
from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.f_measure_paper import f_measure
import torch.nn.functional as F
from logger_utils import setup_logger


# ===================== Boundary Extraction GT =====================

def extract_boundary(mask):
    """mask shape (B,1,H,W), float32"""
    sobel_x = torch.tensor([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1,-2,-1],
                            [0,0,0],
                            [1,2,1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)

    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    return torch.sqrt(gx**2 + gy**2)


# ===================== Segmentation Loss =====================

def dice_loss(pred, gt):
    p = torch.sigmoid(pred)
    num = 2 * (p*gt).sum() + 1e-7
    den = p.sum() + gt.sum() + 1e-7
    return 1 - num/den


def mae(pred, gt):
    p = torch.sigmoid(pred)
    return torch.mean(torch.abs(p - gt))


# ===================== TRAIN =====================

def train_epoch(model, loader, optimizer, scaler, device, is_bem):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    lambda_b = 0.3

    loss_total = 0
    n = 0

    for img, mask_gt in loader:
        img = img.to(device)
        mask_gt = mask_gt.to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            # ----------------- CASE A BASELINE -----------------
            if not is_bem:
                mask_pred = model(img)
                loss = bce(mask_pred, mask_gt) + dice_loss(mask_pred, mask_gt)

            # ----------------- CASE B BEM -----------------
            else:
                mask_pred, boundary_pred = model(img)

                L_seg = bce(mask_pred, mask_gt) + dice_loss(mask_pred, mask_gt)
                GT_b = extract_boundary(mask_gt)
                L_b = F.l1_loss(boundary_pred, GT_b)

                loss = L_seg + lambda_b * L_b

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_total += loss.item()
        n += 1

    return loss_total / n


# ===================== VALIDATE =====================

@torch.no_grad()
def validate(model, loader, device, is_bem):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    res = {"loss":0,"S":0,"MAE":0,"E":0,"F":0}
    n = 0

    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        # ---------- CASE: BEM model ----------
        if is_bem:
            out,_ = model(img)
        else:
            out = model(img)

        loss = bce(out,mask)+dice_loss(out,mask)

        probs = torch.sigmoid(out)
        for i in range(img.shape[0]):
            res["S"]  += s_measure(probs[i], mask[i]).item()
            res["E"]  += e_measure(probs[i], mask[i]).item()
            res["MAE"]+= mae(out[i], mask[i]).item()
            res["F"]  += f_measure(probs[i], mask[i]).item()

        res["loss"]+= loss.item()
        n += img.shape[0]

    return {k: v/n for k,v in res.items()}


def plot_loss(train_loss, val_loss, fpath):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(fpath)
    plt.close()


# ===================== MAIN =====================
def main():

    # ================== CHỌN MODEL ==================
    # model_name = "UNetPP"
    # model_name = "UNetPP_B5"
    # model_name = "UNetPP_B2"
    # model_name = "UNetPP_B3"
    model_name = "UNetPP_B2_BEM_X00"

    root="../MHCD_seg"
    epochs=120
    bs=16
    img_size=256
    lr=1e-4
    device="cuda"

    os.makedirs("logs", exist_ok=True)
    logger = setup_logger("logs", "train.log")

    # ================== Dataset ==================
    train_ds = MHCDDataset(root,"train",img_size,logger=logger)
    val_ds   = MHCDDataset(root,"val",img_size,logger=logger)

    logger.info("===== TRAIN CONFIG =====")
    logger.info(f"Model : {model_name}")
    logger.info(f"Dataset root : {root}")
    logger.info(f"Train images : {len(train_ds)}")
    logger.info(f"Val images   : {len(val_ds)}")
    logger.info(f"Batch size   : {bs}")
    logger.info(f"Epochs       : {epochs}")
    logger.info(f"Image size   : {img_size}")
    logger.info(f"Device       : {device}")

    train_loader = DataLoader(train_ds,bs,shuffle=True,num_workers=4)
    val_loader   = DataLoader(val_ds,bs,shuffle=False,num_workers=4)

    # ================== MODEL SELECT ==================
    if model_name == "UNetPP":
        model = UNetPP().to(device)
        is_bem = False
    elif model_name == "UNetPP_B1":
        model = UNetPP_B1().to(device)
        is_bem = False
    elif model_name == "UNetPP_B2":
        model = UNetPP_B2().to(device)
        is_bem = False
    elif model_name == "UNetPP_B3":
        model = UNetPP_B3().to(device)
        is_bem = False
    elif model_name == "UNetPP_B5":
        model = UNetPP_B5().to(device)
        is_bem = False
    elif model_name == "UNetPP_B2_BEM":
        model = UNetPP_B2_BEM().to(device)
        is_bem = True
    elif model_name == "UNetPP_B3_BEM":
        model = UNetPP_B3_BEM().to(device)
        is_bem = True
    elif model_name == "UNetPP_B2_BEM_X00":
        model = UNetPP_B2_BEM_X00().to(device)
        is_bem = True

    else:
        raise ValueError("model_name invalid")

    opt = torch.optim.AdamW(model.parameters(),lr=lr)
    scaler = GradScaler()

    start_epoch=1
    best_S = 0

    ckpt="logs/last.pth"
    if os.path.exists(ckpt):
        ck = torch.load(ckpt)

        # BEM checkpoint: có optimizer / scaler
        if is_bem and "model" in ck:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch = ck["epoch"]+1
            best_S = ck["bestS"]
        else:
            model.load_state_dict(ck)

        print(f"[Resume] epoch={start_epoch} best_S={best_S}")
        logger.info(f"[Resume] epoch={start_epoch} best_S={best_S}")

    # ================== TRAINING ==================
    train_losses=[]
    val_losses=[]

    log_csv="logs/train_log.csv"
    if not os.path.exists(log_csv):
        with open(log_csv,"w",newline="") as f:
            csv.writer(f).writerow(["epoch","train","val","S","MAE","E","F"])

    for e in range(start_epoch,epochs+1):
        print(f"=== Epoch {e}/{epochs} ===")
        logger.info(f"[EPOCH {e}] START")
        
        tr = train_epoch(model,train_loader,opt,scaler,device,is_bem)
        vc = validate(model,val_loader,device,is_bem)

        train_losses.append(tr)
        val_losses.append(vc["loss"])

        print(f"[Train] {tr:.4f}")
        print(f"[Val] loss={vc['loss']:.4f}  S={vc['S']:.4f}  MAE={vc['MAE']:.4f}  E={vc['E']:.4f}  F={vc['F']:.4f}")

        # Save
        with open(log_csv,"a",newline="") as f:
            csv.writer(f).writerow([e,tr,vc['loss'],vc['S'],vc['MAE'],vc['E'],vc['F']])

        torch.save({
            "epoch":e,
            "model":model.state_dict(),
            "optimizer":opt.state_dict(),
            "scaler":scaler.state_dict(),
            "bestS":best_S
        },"logs/last.pth")

        if vc["S"] > best_S:
            best_S = vc["S"]
            torch.save(model.state_dict(),"logs/best_S.pth")
            print(f"🔥 New best S={best_S:.4f}")

    plot_loss(train_losses,val_losses,"logs/loss_curve.png")


if __name__=="__main__":
    main()

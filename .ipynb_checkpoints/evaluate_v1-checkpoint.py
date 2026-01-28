import os
import torch
from torch.utils.data import DataLoader

from datasets.mhcd_dataset import MHCDDataset
from models.unetpp_bem import UNetPP, UNetPP_B1, UNetPP_B2, UNetPP_B3, UNetPP_B5, UNetPP_B2_BEM, UNetPP_B3_BEM, UNetPP_B2_BEM_X00
from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.fweighted_measure import fw_measure
from logger_utils import setup_logger

# =========================================================
# Helper: load checkpoint smart
# =========================================================
def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location=device)

    # case B: saved as raw state_dict
    if any(k.startswith("encoder") or k.startswith("decoder") for k in ckpt.keys()):
        model.load_state_dict(ckpt)
        return

    # case A: saved as dict
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        return

    raise RuntimeError(f"Invalid checkpoint format: {path}")


@torch.no_grad()
def eval_test():

    # ============== Config lựa chọn model =================
    #model_name = "UNetPP"
    # model_name = "UNetPP_B2"
    # model_name = "UNetPP_B5"
    model_name = "UNetPP_B2_BEM_X00"

    ckpt_path = "logs/best_S.pth"
    batch_size = 8
    root = "../MHCD_seg"
    device = "cuda"

    logger = setup_logger("logs", "test.log")
    logger.info("==================== TEST MODE ====================")
    logger.info(f"Model     : {model_name}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Dataset   : {root}")

    # ===================== Model init =====================
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
        raise ValueError("Invalid model_name")

    # ================== Load checkpoint ===================
    #load_ckpt(model, ckpt_path, device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # ================== Dataset ===========================
    test_set = MHCDDataset(root, "test", logger=logger)
    test_loader = DataLoader(test_set, batch_size, False)

    logger.info(f"Test images = {len(test_set)}")

    # ================== Metrics ===========================
    S=0; E=0; MAE=0; Fw=0
    n=0

    logger.info("=============== TEST START ===============")

    for img,mask in test_loader:
        img = img.to(device)
        mask = mask.to(device)

        # --------- BEM ----------
        if is_bem:
            out,_ = model(img)
        else:
            out = model(img)

        p = torch.sigmoid(out)

        for i in range(img.shape[0]):
            S += s_measure (p[i], mask[i]).item()
            E += e_measure (p[i], mask[i]).item()
            MAE += torch.abs(p[i]-mask[i]).mean().item()
            Fw += fw_measure(p[i], mask[i]).item()
            n+=1

    # ================= Print ======================
    print("\n========== TEST RESULT ==========\n")
    print(f" S-measure  : {S/n:.4f}")
    print(f" E-measure  : {E/n:.4f}")
    print(f" Fw-measure : {Fw/n:.4f}")
    print(f" MAE        : {MAE/n:.4f}")

    # ================= Log =======================
    logger.info("=============== TEST RESULTS ===============")
    logger.info(f"S-measure  = {S/n:.4f}")
    logger.info(f"E-measure  = {E/n:.4f}")
    logger.info(f"Weighted F = {Fw/n:.4f}")
    logger.info(f"MAE        = {MAE/n:.4f}")
    logger.info("================= TEST END =================\n")


if __name__=="__main__":
    eval_test()

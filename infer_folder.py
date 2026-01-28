"""
Batch inference on a folder of images for COD models.
"""

import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import (
    UNet, UNet_B3, UNet_BEM,
    UNetPP, UNetPP_B3, UNetPP_BEM,
    UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM,
)

MODEL_REGISTRY = {
    "UNet": UNet, "UNet_B3": UNet_B3, "UNet_BEM": UNet_BEM,
    "UNetPP": UNetPP, "UNetPP_B3": UNetPP_B3, "UNetPP_BEM": UNetPP_BEM,
    "UNet3Plus": UNet3Plus, "UNet3Plus_B3": UNet3Plus_B3, "UNet3Plus_BEM": UNet3Plus_BEM,
}


def load_model(model_name, weight_path, device="cuda"):
    model = MODEL_REGISTRY[model_name](n_classes=1).to(device)
    ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def build_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def infer_folder(model, input_folder, save_folder="infer_out", img_size=640,
                 device="cuda", exts=(".jpg", ".png", ".jpeg", ".bmp")):
    os.makedirs(save_folder, exist_ok=True)
    transform = build_transform(img_size)

    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(exts)
    ]
    if not files:
        print(f"No valid images found in: {input_folder}")
        return

    print(f"Processing {len(files)} images...")
    t0 = time.time()

    for path in tqdm(files, desc="Inference"):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Skipping: {path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = transform(image=img_rgb)
        tensor = t["image"].unsqueeze(0).to(device)

        output = model(tensor)
        # BEM models return (mask, boundary)
        mask_logits = output[0] if isinstance(output, tuple) else output
        prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()

        mask255 = (prob * 255).astype(np.uint8)

        # Restore original size
        h, w = img.shape[:2]
        mask_resized = cv2.resize(mask255, (w, h), interpolation=cv2.INTER_NEAREST)

        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(save_folder, f"{name}_mask.png")
        cv2.imwrite(out_path, mask_resized)

    print(f"Done. Masks saved to: {save_folder}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch folder inference")
    parser.add_argument("--model", type=str, default="UNetPP_BEM",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--out", type=str, default="infer_out")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    model = load_model(args.model, args.weights, args.device)
    infer_folder(model, args.input_folder, save_folder=args.out,
                 img_size=args.img_size, device=args.device)

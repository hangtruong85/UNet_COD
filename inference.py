"""
Single image inference for COD models.
"""

import os
import cv2
import torch
import numpy as np
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


def preprocess_image(img_path, img_size=640):
    image = cv2.imread(img_path)
    assert image is not None, f"Cannot read image: {img_path}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    t = transform(image=image)
    return t["image"].unsqueeze(0)


def inference_single(model, img_path, save_dir="infer_out", img_size=640, device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    tensor = preprocess_image(img_path, img_size).to(device)

    with torch.no_grad():
        output = model(tensor)
        # BEM models return (mask, boundary)
        mask_logits = output[0] if isinstance(output, tuple) else output
        mask_prob = torch.sigmoid(mask_logits)[0, 0]

    mask = (mask_prob.cpu().numpy() * 255).astype(np.uint8)

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(save_dir, base + "_mask.png")
    cv2.imwrite(out_path, mask)

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single image inference")
    parser.add_argument("--model", type=str, default="UNetPP_BEM",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="infer_out")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    model = load_model(args.model, args.weights, args.device)
    result = inference_single(model, args.image, args.out, args.img_size, args.device)
    print(f"Saved: {result}")

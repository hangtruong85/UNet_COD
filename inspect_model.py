"""
Inspect model structure to understand layer hierarchy.
Useful for debugging and understanding model architectures.
"""

import torch
import torch.nn as nn
from models import (
    UNet, UNet_B3, UNet_BEM,
    UNetPP, UNetPP_B3, UNetPP_BEM,
    UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM,
)


def print_model_structure(model, model_name, max_depth=3):
    print("\n" + "=" * 80)
    print(f"MODEL STRUCTURE: {model_name}")
    print("=" * 80 + "\n")

    def print_module(module, prefix="", depth=0):
        if depth > max_depth:
            return

        module_type = type(module).__name__
        num_params = sum(p.numel() for p in module.parameters(recurse=False))

        if num_params > 0:
            print(f"{prefix}[{module_type}] {num_params:,} params")
        else:
            print(f"{prefix}[{module_type}]")

        children = list(module.named_children())
        if children and depth < max_depth:
            for i, (name, child) in enumerate(children):
                is_last = (i == len(children) - 1)
                connector = "  " if is_last else "| "
                print(f"{prefix}{'+-' if not is_last else '\\-'} {name}:")
                print_module(child, prefix + connector, depth + 1)

    print_module(model)
    print()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def find_feature_layers(model, model_name):
    print(f"Feature extraction layers for: {model_name}")
    print("-" * 60)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 16 <= module.out_channels <= 512:
            print(f"  {name:<50s} Conv2d({module.in_channels} -> {module.out_channels}, k={module.kernel_size})")


def main():
    print("=" * 80)
    print("MODEL STRUCTURE INSPECTOR")
    print("=" * 80)

    models = {
        'UNet_B3':       UNet_B3(n_classes=1),
        'UNet_BEM':      UNet_BEM(n_classes=1),
        'UNetPP_B3':     UNetPP_B3(n_classes=1),
        'UNetPP_BEM':    UNetPP_BEM(n_classes=1),
        'UNet3Plus_B3':  UNet3Plus_B3(n_classes=1),
        'UNet3Plus_BEM': UNet3Plus_BEM(n_classes=1),
    }

    for name, model in models.items():
        total, trainable = count_parameters(model)
        print(f"\n{'=' * 80}")
        print(f"{name}: {total:,} total params ({trainable:,} trainable)")
        print(f"{'=' * 80}")

        print_model_structure(model, name, max_depth=2)
        find_feature_layers(model, name)

        # Test forward pass
        x = torch.randn(1, 3, 352, 352)
        model.eval()
        with torch.no_grad():
            out = model(x)
        if isinstance(out, tuple):
            print(f"\nOutput: mask={out[0].shape}, boundary={out[1].shape}")
        else:
            print(f"\nOutput: {out.shape}")


if __name__ == "__main__":
    main()

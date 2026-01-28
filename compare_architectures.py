"""
Compare UNet vs UNet++ vs UNet3+ architectures.
Measures parameter count, FLOPs, and inference time.
"""

import os
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from models import (
    UNet, UNet_B3, UNet_BEM,
    UNetPP, UNetPP_B3, UNetPP_BEM,
    UNet3Plus, UNet3Plus_B3, UNet3Plus_BEM,
)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, input_size=(1, 3, 352, 352)):
    try:
        from thop import profile
        x = torch.randn(input_size)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return flops
    except ImportError:
        return None


def measure_inference_time(model, input_size=(1, 3, 352, 352), num_runs=100, device='cuda'):
    model = model.to(device).eval()
    x = torch.randn(input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / num_runs
    else:
        import time
        with torch.no_grad():
            t0 = time.time()
            for _ in range(num_runs):
                _ = model(x)
            elapsed_time = (time.time() - t0) / num_runs * 1000

    return elapsed_time


def compare_models():
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: UNet vs UNet++ vs UNet3+")
    print("=" * 80 + "\n")

    models_config = {
        # Baseline
        "UNet_B3":       UNet_B3,
        "UNetPP_B3":     UNetPP_B3,
        "UNet3Plus_B3":  UNet3Plus_B3,
        # With BEM
        "UNet_BEM":      UNet_BEM,
        "UNetPP_BEM":    UNetPP_BEM,
        "UNet3Plus_BEM": UNet3Plus_BEM,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    results = []

    for name, model_cls in models_config.items():
        print(f"Analyzing: {name}...")
        try:
            model = model_cls(n_classes=1)
            total_params, trainable_params = count_parameters(model)
            flops = estimate_flops(model)

            inf_time = None
            if device == 'cuda':
                inf_time = measure_inference_time(model, device=device)

            result = {
                'Model': name,
                'Params (M)': total_params / 1e6,
                'FLOPs (G)': flops / 1e9 if flops else None,
                'Inference (ms)': inf_time,
            }
            results.append(result)

            print(f"  Params: {result['Params (M)']:.2f}M")
            if flops:
                print(f"  FLOPs: {result['FLOPs (G)']:.2f}G")
            if inf_time:
                print(f"  Inference: {inf_time:.2f}ms")
        except Exception as e:
            print(f"  Error: {e}")

    # Print table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Params (M)':>12} {'FLOPs (G)':>12} {'Inf (ms)':>12}")
    print("-" * 56)
    for r in results:
        flops_str = f"{r['FLOPs (G)']:.2f}" if r['FLOPs (G)'] else "N/A"
        inf_str = f"{r['Inference (ms)']:.2f}" if r['Inference (ms)'] else "N/A"
        print(f"{r['Model']:<20} {r['Params (M)']:>12.2f} {flops_str:>12} {inf_str:>12}")
    print("=" * 80)

    # Save results
    os.makedirs("comparison_results", exist_ok=True)
    with open("comparison_results/model_complexity.json", 'w') as f:
        json.dump(results, f, indent=4, default=str)

    # Generate plots
    _generate_plots(results)

    return results


def _generate_plots(results):
    baseline = [r for r in results if 'BEM' not in r['Model']]
    bem = [r for r in results if 'BEM' in r['Model']]

    if not baseline or not bem:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(baseline))
    width = 0.35

    # Parameters
    axes[0].bar(x - width / 2, [r['Params (M)'] for r in baseline], width, label='Baseline', alpha=0.8)
    axes[0].bar(x + width / 2, [r['Params (M)'] for r in bem], width, label='+ BEM', alpha=0.8)
    axes[0].set_ylabel('Parameters (M)')
    axes[0].set_title('Model Size')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['UNet', 'UNet++', 'UNet3+'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FLOPs
    b_flops = [r['FLOPs (G)'] for r in baseline if r['FLOPs (G)']]
    bem_flops = [r['FLOPs (G)'] for r in bem if r['FLOPs (G)']]
    if b_flops and bem_flops:
        axes[1].bar(x - width / 2, b_flops, width, label='Baseline', alpha=0.8)
        axes[1].bar(x + width / 2, bem_flops, width, label='+ BEM', alpha=0.8)
        axes[1].set_ylabel('FLOPs (G)')
        axes[1].set_title('Computational Cost')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['UNet', 'UNet++', 'UNet3+'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_results/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: comparison_results/model_comparison.png")


if __name__ == "__main__":
    compare_models()

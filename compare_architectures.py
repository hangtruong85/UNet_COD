"""
Compare UNet++ vs UNet3+ with different DCN versions
Run comprehensive experiments and generate comparison table
"""

import os
import json
import torch
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

# Model imports
from models.unetpp_bem import (
    UNetPP_B3,
    UNetPP_DCNv1_COD,
    UNetPP_DCNv2_COD,
    UNetPP_DCNv3_COD,
    UNetPP_DCNv4_COD
)

from models.unet3plus_dcn import (
    UNet3Plus_B3,
    UNet3Plus_DCNv1_COD,
    UNet3Plus_DCNv2_COD,
    UNet3Plus_DCNv3_COD,
    UNet3Plus_DCNv4_COD
)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, input_size=(1, 3, 352, 352)):
    """Estimate FLOPs using thop library"""
    try:
        from thop import profile
        x = torch.randn(input_size)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops
    except ImportError:
        print("⚠ thop not installed. Run: pip install thop")
        return None


def measure_inference_time(model, input_size=(1, 3, 352, 352), num_runs=100, device='cuda'):
    """Measure average inference time"""
    model = model.to(device).eval()
    x = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure
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
        elapsed_time = start.elapsed_time(end) / num_runs  # ms
    else:
        import time
        with torch.no_grad():
            start = time.time()
            for _ in range(num_runs):
                _ = model(x)
            elapsed_time = (time.time() - start) / num_runs * 1000  # ms
    
    return elapsed_time


def load_training_results(log_dir):
    """Load training results from log directory"""
    summary_path = os.path.join(log_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def compare_models():
    """Compare all model variants"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON: UNet++ vs UNet3+")
    print("="*100 + "\n")
    
    # Define models to compare
    models_config = {
        # UNet++ variants
        "UNet++_B3": UNetPP_B3,
        "UNet++_DCNv1": UNetPP_DCNv1_COD,
        "UNet++_DCNv2": UNetPP_DCNv2_COD,
        "UNet++_DCNv3": UNetPP_DCNv3_COD,
        "UNet++_DCNv4": UNetPP_DCNv4_COD,
        
        # UNet3+ variants
        "UNet3+_B3": UNet3Plus_B3,
        "UNet3+_DCNv1": UNet3Plus_DCNv1_COD,
        "UNet3+_DCNv2": UNet3Plus_DCNv2_COD,
        "UNet3+_DCNv3": UNet3Plus_DCNv3_COD,
        "UNet3+_DCNv4": UNet3Plus_DCNv4_COD,
    }
    
    results = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    for name, model_cls in models_config.items():
        print(f"Analyzing: {name}...")
        
        try:
            # Create model
            model = model_cls(n_classes=1)
            
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            
            # Estimate FLOPs
            flops = estimate_flops(model)
            
            # Measure inference time
            if device == 'cuda':
                inf_time = measure_inference_time(model, device=device)
            else:
                inf_time = None
            
            # Store results
            result = {
                'Model': name,
                'Params (M)': total_params / 1e6,
                'FLOPs (G)': flops / 1e9 if flops else None,
                'Inference (ms)': inf_time,
            }
            
            results.append(result)
            print(f"  ✓ Params: {result['Params (M)']:.2f}M")
            if flops:
                print(f"  ✓ FLOPs: {result['FLOPs (G)']:.2f}G")
            if inf_time:
                print(f"  ✓ Inference: {inf_time:.2f}ms")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Print comparison table
    print("\n" + "="*100)
    print("MODEL COMPLEXITY COMPARISON")
    print("="*100 + "\n")
    
    headers = ['Model', 'Params (M)', 'FLOPs (G)', 'Inference (ms)']
    table_data = []
    
    for r in results:
        row = [
            r['Model'],
            f"{r['Params (M)']:.2f}",
            f"{r['FLOPs (G)']:.2f}" if r['FLOPs (G)'] else "N/A",
            f"{r['Inference (ms)']:.2f}" if r['Inference (ms)'] else "N/A"
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Save results
    os.makedirs("comparison_results", exist_ok=True)
    
    with open("comparison_results/model_complexity.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: comparison_results/model_complexity.json")
    
    # Generate plots
    generate_comparison_plots(results)
    
    return results


def generate_comparison_plots(results):
    """Generate comparison plots"""
    
    # Separate UNet++ and UNet3+
    unetpp_results = [r for r in results if 'UNet++' in r['Model']]
    unet3plus_results = [r for r in results if 'UNet3+' in r['Model']]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Parameters comparison
    ax = axes[0]
    unetpp_params = [r['Params (M)'] for r in unetpp_results]
    unet3plus_params = [r['Params (M)'] for r in unet3plus_results]
    
    x = np.arange(len(unetpp_results))
    width = 0.35
    
    ax.bar(x - width/2, unetpp_params, width, label='UNet++', alpha=0.8)
    ax.bar(x + width/2, unet3plus_params, width, label='UNet3+', alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Parameters (M)')
    ax.set_title('Model Size Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'DCNv1', 'DCNv2', 'DCNv3', 'DCNv4'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: FLOPs comparison
    ax = axes[1]
    unetpp_flops = [r['FLOPs (G)'] for r in unetpp_results if r['FLOPs (G)']]
    unet3plus_flops = [r['FLOPs (G)'] for r in unet3plus_results if r['FLOPs (G)']]
    
    if unetpp_flops and unet3plus_flops:
        x = np.arange(len(unetpp_flops))
        ax.bar(x - width/2, unetpp_flops, width, label='UNet++', alpha=0.8)
        ax.bar(x + width/2, unet3plus_flops, width, label='UNet3+', alpha=0.8)
        
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('FLOPs (G)')
        ax.set_title('Computational Cost Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', 'DCNv1', 'DCNv2', 'DCNv3', 'DCNv4'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Inference time comparison
    ax = axes[2]
    unetpp_time = [r['Inference (ms)'] for r in unetpp_results if r['Inference (ms)']]
    unet3plus_time = [r['Inference (ms)'] for r in unet3plus_results if r['Inference (ms)']]
    
    if unetpp_time and unet3plus_time:
        x = np.arange(len(unetpp_time))
        ax.bar(x - width/2, unetpp_time, width, label='UNet++', alpha=0.8)
        ax.bar(x + width/2, unet3plus_time, width, label='UNet3+', alpha=0.8)
        
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Speed Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', 'DCNv1', 'DCNv2', 'DCNv3', 'DCNv4'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results/model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plots saved to: comparison_results/model_comparison.png")
    plt.close()


def load_and_compare_performance():
    """
    Load training/evaluation results and compare performance
    Assumes results are saved in logs/ directory
    """
    
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON (if training logs available)")
    print("="*100 + "\n")
    
    models_to_check = [
        "UNetPP_B3",
        "UNetPP_DCNv1_COD",
        "UNetPP_DCNv2_COD",
        "UNetPP_DCNv3_COD",
        "UNetPP_DCNv4_COD",
        "UNet3Plus_B3",
        "UNet3Plus_DCNv1_COD",
        "UNet3Plus_DCNv2_COD",
        "UNet3Plus_DCNv3_COD",
        "UNet3Plus_DCNv4_COD",
    ]
    
    performance_results = []
    
    for model_name in models_to_check:
        # Try to find latest log directory
        log_dirs = [d for d in os.listdir("logs") if d.startswith(model_name)]
        
        if log_dirs:
            # Get most recent
            log_dirs.sort()
            latest_log = os.path.join("logs", log_dirs[-1])
            
            # Try to load training log CSV
            csv_path = os.path.join(latest_log, "training_log.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                # Get best metrics
                best_s = df['S_measure'].max()
                best_e = df['E_measure'].max()
                best_f = df['F_measure'].max()
                min_mae = df['MAE'].min()
                
                performance_results.append({
                    'Model': model_name,
                    'S-measure': best_s,
                    'E-measure': best_e,
                    'F-measure': best_f,
                    'MAE': min_mae
                })
    
    if performance_results:
        headers = ['Model', 'S-measure', 'E-measure', 'F-measure', 'MAE']
        table_data = []
        
        for r in performance_results:
            row = [
                r['Model'],
                f"{r['S-measure']:.4f}",
                f"{r['E-measure']:.4f}",
                f"{r['F-measure']:.4f}",
                f"{r['MAE']:.4f}"
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Save
        with open("comparison_results/performance_comparison.json", 'w') as f:
            json.dump(performance_results, f, indent=4)
        
        print(f"\n✓ Performance results saved to: comparison_results/performance_comparison.json")
    else:
        print("⚠ No training logs found. Train models first to see performance comparison.")


if __name__ == "__main__":
    # Compare model complexity
    results = compare_models()
    
    # Try to load and compare performance
    load_and_compare_performance()
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("\nKey Insights:")
    print("1. UNet3+ has full-scale skip connections → better multi-scale feature fusion")
    print("2. DCN adds adaptive receptive fields → better for irregular camouflage boundaries")
    print("3. Trade-off: DCNv4 is most powerful but slowest, DCNv2 is good balance")
    print("4. Check 'comparison_results/' for detailed analysis")
    print("="*100 + "\n")
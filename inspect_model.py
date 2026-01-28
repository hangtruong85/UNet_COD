"""
Inspect model structure to understand layer hierarchy
Useful for debugging visualization hooks
"""

import torch
import torch.nn as nn
from models.unetpp_bem import UNetPP_B3, UNetPP_DCNv3_COD
from models.unet3plus_dcn import UNet3Plus_B3, UNet3Plus_DCNv3_COD


def print_model_structure(model, model_name, max_depth=4):
    """
    Print hierarchical model structure
    Args:
        model: PyTorch model
        model_name: Name for display
        max_depth: Maximum depth to print
    """
    print("\n" + "="*80)
    print(f"MODEL STRUCTURE: {model_name}")
    print("="*80 + "\n")
    
    def print_module(module, prefix="", depth=0):
        if depth > max_depth:
            return
        
        # Get module type
        module_type = type(module).__name__
        
        # Count parameters
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        
        # Print current module
        if num_params > 0:
            print(f"{prefix}├─ [{module_type}] {num_params:,} params")
        else:
            print(f"{prefix}├─ [{module_type}]")
        
        # Print children
        children = list(module.named_children())
        if children and depth < max_depth:
            for i, (name, child) in enumerate(children):
                is_last = (i == len(children) - 1)
                connector = "└─" if is_last else "├─"
                extension = "   " if is_last else "│  "
                
                print(f"{prefix}{connector} {name}:")
                print_module(child, prefix + extension, depth + 1)
    
    print_module(model)
    print("\n" + "="*80 + "\n")


def find_feature_layers(model, model_name):
    """
    Find suitable layers for feature extraction
    """
    print(f"Finding feature extraction layers for: {model_name}")
    print("-"*80)
    
    suitable_layers = []
    
    for name, module in model.named_modules():
        # Look for conv layers with reasonable channel sizes
        if isinstance(module, nn.Conv2d):
            in_ch = module.in_channels
            out_ch = module.out_channels
            
            # Skip very small or very large layers
            if 16 <= out_ch <= 512:
                suitable_layers.append({
                    'path': name,
                    'type': 'Conv2d',
                    'in_channels': in_ch,
                    'out_channels': out_ch,
                    'kernel_size': module.kernel_size
                })
        
        # Look for sequential blocks
        elif isinstance(module, nn.Sequential) and len(list(module.children())) > 2:
            # Check if it contains conv layers
            has_conv = any(isinstance(m, nn.Conv2d) for m in module.children())
            if has_conv:
                suitable_layers.append({
                    'path': name,
                    'type': 'Sequential',
                    'num_layers': len(list(module.children()))
                })
    
    # Print suitable layers
    print(f"\nFound {len(suitable_layers)} suitable layers for feature extraction:\n")
    
    for i, layer in enumerate(suitable_layers[:20], 1):  # Show first 20
        if layer['type'] == 'Conv2d':
            print(f"{i:2d}. {layer['path']:<50s} "
                  f"Conv2d({layer['in_channels']} → {layer['out_channels']}, "
                  f"k={layer['kernel_size']})")
        else:
            print(f"{i:2d}. {layer['path']:<50s} "
                  f"Sequential({layer['num_layers']} layers)")
    
    if len(suitable_layers) > 20:
        print(f"\n... and {len(suitable_layers) - 20} more layers")
    
    print("\n" + "="*80 + "\n")
    
    return suitable_layers


def test_hook_registration(model, layer_path):
    """
    Test if we can register a hook at the given path
    """
    try:
        # Navigate to module
        module = model
        for attr in layer_path.split('.'):
            module = getattr(module, attr)
        
        # Try to register hook
        def test_hook(module, input, output):
            print(f"  ✓ Hook triggered! Output shape: {output.shape}")
        
        hook = module.register_forward_hook(test_hook)
        
        # Test forward pass
        x = torch.randn(1, 3, 352, 352)
        with torch.no_grad():
            _ = model(x)
        
        # Remove hook
        hook.remove()
        
        return True
    
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def suggest_visualization_config(model, model_name):
    """
    Suggest best layers for visualization
    """
    print(f"\nSUGGESTED VISUALIZATION CONFIG for {model_name}:")
    print("-"*80)
    
    layers = find_feature_layers(model, model_name)
    
    # Filter for encoder layers
    encoder_layers = [l for l in layers if 'encoder' in l['path'].lower()]
    decoder_layers = [l for l in layers if 'decoder' in l['path'].lower()]
    
    print("\nRecommended Encoder Layers:")
    for i, layer in enumerate(encoder_layers[:5], 1):
        print(f"  {i}. {layer['path']}")
    
    print("\nRecommended Decoder Layers:")
    for i, layer in enumerate(decoder_layers[:5], 1):
        print(f"  {i}. {layer['path']}")
    
    print("\nPython Dict Format:")
    print("layer_names = {")
    
    for i, layer in enumerate(encoder_layers[:3]):
        print(f"    'encoder_{i}': '{layer['path']}',")
    
    for i, layer in enumerate(decoder_layers[:3]):
        print(f"    'decoder_{i}': '{layer['path']}',")
    
    print("}")
    print("\n" + "="*80 + "\n")


def main():
    """Main inspection"""
    
    print("\n" + "="*100)
    print("MODEL STRUCTURE INSPECTOR")
    print("="*100)
    
    models = {
        'UNetPP_B3': UNetPP_B3(n_classes=1),
        'UNetPP_DCNv3': UNetPP_DCNv3_COD(n_classes=1),
        'UNet3Plus_B3': UNet3Plus_B3(n_classes=1),
        'UNet3Plus_DCNv3': UNet3Plus_DCNv3_COD(n_classes=1),
    }
    
    for name, model in models.items():
        # Print structure
        print_model_structure(model, name, max_depth=3)
        
        # Find suitable layers
        layers = find_feature_layers(model, name)
        
        # Suggest config
        suggest_visualization_config(model, name)
        
        # Test a few hooks
        print(f"\nTesting hook registration for {name}:")
        print("-"*80)
        
        # Test first encoder layer
        encoder_layers = [l for l in layers if 'encoder' in l['path'].lower()]
        if encoder_layers:
            print(f"\nTesting encoder layer: {encoder_layers[0]['path']}")
            test_hook_registration(model, encoder_layers[0]['path'])
        
        # Test first decoder layer
        decoder_layers = [l for l in layers if 'decoder' in l['path'].lower()]
        if decoder_layers:
            print(f"\nTesting decoder layer: {decoder_layers[0]['path']}")
            test_hook_registration(model, decoder_layers[0]['path'])
        
        print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
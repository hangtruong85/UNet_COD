
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block for ResNet
    Implements: output = F(x) + x
    where F(x) is two weight layers with ReLU activation
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # First weight layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second weight layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer for identity mapping when dimensions change
        self.downsample = downsample
        
    def forward(self, x):
        # Save input for skip connection
        identity = x
        
        # F(x): First weight layer + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # F(x): Second weight layer (no ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # If dimensions changed, adjust identity
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection: F(x) + x
        out += identity
        
        # Final ReLU activation
        out = self.relu(out)
        
        return out


# Example usage
if __name__ == "__main__":
    # Create a residual block
    block = ResidualBlock(in_channels=64, out_channels=64)
    
    # Test with random input
    x = torch.randn(1, 64, 56, 56)  # Batch=1, Channels=64, H=56, W=56
    output = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Example with dimension change (need downsample)
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128)
    )
    block_with_downsample = ResidualBlock(64, 128, stride=2, downsample=downsample)
    
    output2 = block_with_downsample(x)
    print(f"\nWith downsample:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output2.shape}")

import torch.nn as nn
import torch

from typing import *

class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d


        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()


    def forward(self, x, return_features=False):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        if return_features:
            return x

        x = self.output_layer(x)
        
        return x

class SimpleCNN(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, output_d: int, img_size: int, 
                 conv_stride: int = 2, kernel_size: int = 5):
        super(SimpleCNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.output_d = output_d
        self.conv_stride = conv_stride

        # --- Layer Definitions ---
        # Layer 1: Conv with configurable stride + Pool
        padding = kernel_size // 2
        self.layer1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, 
                                stride=conv_stride, padding=padding)
        # Layer 2: Conv with configurable stride + Pool
        self.layer2 = nn.Conv2d(hidden_channels, hidden_channels, 
                                kernel_size=kernel_size, stride=conv_stride, padding=padding)
        
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate total spatial reduction: (stride * pool) for each layer
        spatial_reduction = (conv_stride * 2) ** 2  # Each layer: stride*2 (conv+pool)
        final_size = int(img_size / (conv_stride * 2 * conv_stride * 2))

        # Final classification layer
        self.output_layer = nn.Linear(hidden_channels * (final_size * final_size), output_d)


    def forward(self, x):        
        # Layer 1: Conv(stride=2) -> ReLU -> Pool
        x = self.layer1(x)
        x = self.activation(x)
        x = self.pool(x)

        # Layer 2: Conv(stride=2) -> ReLU -> Pool
        x = self.layer2(x)
        x = self.activation(x)
        x = self.pool(x)

        # Flatten (Batch, Flattened_Size)
        x = x.view(x.shape[0], -1) 

        x = self.output_layer(x)
        
        return x
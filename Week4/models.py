
import torch.nn as nn
import torch
import copy
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt

from typing import *

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.v2  as T
import numpy as np 

import os
import pdb

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.early_stop = False
        self.counter = 0
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.patience != -1:
            self.counter += 1
            if  self.counter == self.patience:
                self.early_stop = True
                
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        
        # Step 1: Reshape into (N, groups, channels_per_group, H, W)
        # This prepares the tensor for the transposition
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Step 2: Transpose (swap) the group and channel-per-group dimensions
        # This is the 'shuffle' step
        x = x.transpose(1, 2).contiguous()
        
        # Step 3: Flatten back to (N, C, H, W) for the next convolution
        x = x.view(batch_size, num_channels, height, width)
        
        return x
                
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=5):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # Global Average Pool
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() # Weight between 0 and 1
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        # Squeeze
        chan_weights = self.squeeze(x).view(batch_size, num_channels)
        # Excitation
        chan_weights = self.excitation(chan_weights).view(batch_size, num_channels, 1, 1)
        # Scale the original feature maps
        return x * chan_weights
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        # Ensure kernel size is odd for symmetric padding
        assert kernel_size in [3, 7], "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        # This convolution compresses 2 channels (avg and max) into 1 spatial mask
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Generate Channel-wise Average and Max Pooling
        # x has shape: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        
        # 2. Concatenate the two descriptors
        # res has shape: (B, 2, H, W)
        res = torch.cat([avg_out, max_out], dim=1)
        
        # 3. Apply convolution and Sigmoid to get the 2D mask
        res = self.conv(res)
        mask = self.sigmoid(res)
        
        # 4. Multiply the original input by the spatial mask
        return x * mask

class MCV_Block(nn.Module):
    def __init__(self, in_f, out_f, model_cfg, first, last, dropout_prob_1=0.0):
        super(MCV_Block, self).__init__()
        g = 5 if in_f % 5 == 0 and out_f % 5 == 0 else 4
        
        if model_cfg["use_dw"] and not first:
            # Depthwise convolutions
            layers = [
                nn.Conv2d(in_f, in_f, kernel_size=3, padding=1, stride=1, groups=in_f, bias=not model_cfg["use_bn"]), # Convolutions applied per-channel
                nn.BatchNorm2d(in_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                nn.ReLU() if not model_cfg["use_prelu"] else nn.PReLU(num_parameters=1), # Activation
                nn.Conv2d(in_f, out_f, kernel_size=1, groups=g if model_cfg["use_shuffle"] and not last else 1, bias=not model_cfg["use_bn"]), # 1x1 convolution to mix channels
                nn.BatchNorm2d(out_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                ChannelShuffle(groups=g) if model_cfg["use_shuffle"] and not last else nn.Identity(), # Shuffle
                nn.ReLU() if not model_cfg["use_prelu"] else nn.PReLU(num_parameters=1) # Activation
            ]
        else:
            # Traditional convolution for the first layer and when not using depthwise convolutions
            layers = [
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1, bias=not model_cfg["use_bn"]), # Set bias to False when having a bn layer
                nn.BatchNorm2d(out_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                nn.ReLU() if not model_cfg["use_prelu"] else nn.PReLU(num_parameters=1) # Activation
            ]
        
        # Dropout after convolutions
        if dropout_prob_1 > 0:
            layers.append(nn.Dropout2d(p=dropout_prob_1))
        
        # Attention layer
        #if model_cfg["use_attention"]: layers.append(SpatialAttention())
        if model_cfg["use_attention"]: layers.append(SEBlock(out_f))
        
        # Construct the main path
        self.block = nn.Sequential(*layers)
        
        # Conv pool reduces activation maps using convolutions with stride = 2
        if model_cfg["use_conv_pool"]:
            self.pool = nn.Sequential(
                            nn.Conv2d(out_f, out_f, kernel_size=3, padding=1, stride=2, groups=out_f, bias=not model_cfg["use_bn"]), # Convolutions applied per-channel
                            nn.BatchNorm2d(out_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                            nn.ReLU(), # Activation
                            nn.Conv2d(out_f, out_f, kernel_size=1, bias=not model_cfg["use_bn"]), # 1x1 convolution to mix channels
                            nn.BatchNorm2d(out_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                            nn.ReLU() # Activation
                        ) if not last or not model_cfg["use_gap"] else nn.Identity()
        # MaxPool only applied in the last block if there is not GAP layer before FC
        else:
            self.pool = nn.MaxPool2d(2) if model_cfg["use_pool"] and (not model_cfg["use_gap"] or not last) else nn.Identity()
        
        # Residual connection
        self.shortcut = nn.Identity()
        self.use_residual = model_cfg["use_residual"]
        if model_cfg["use_residual"] and in_f != out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_f) if model_cfg["use_bn"] else nn.Identity()
            )

    def forward(self, x):
        out = self.block(x)
        
        if self.use_residual:
            out = out + self.shortcut(x)
            out = F.relu(out)
            
        out = self.pool(out)
            
        return out
    
class ClassificationHead(nn.Module):
    def __init__(self, flat_features, num_classes, model_cfg, units_fc, dropout_prob_2=0.0):
        super(ClassificationHead, self).__init__()
        
        if model_cfg["use_gap"]: 
            layers = [
                nn.Flatten(),
            ]
            if dropout_prob_2 > 0:
                layers.append(nn.Dropout(p=dropout_prob_2))
            layers.append(nn.Linear(flat_features, num_classes))
            self.layers = nn.Sequential(*layers)          
        else:
            layers = [
                nn.Flatten(),
                nn.Linear(flat_features, units_fc),
                nn.ReLU(),
            ]
            if dropout_prob_2 > 0:
                layers.append(nn.Dropout(p=dropout_prob_2))
            layers.append(nn.Linear(units_fc, num_classes))
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class MCV_Net(nn.Module):    
    def __init__(self, image_size: int, block_type: str = "baseline", num_classes: int = 8, init_chan: int = 20, 
                 num_blocks: int = 4, filters: list = [12,24,48,96,128], blocks_to_keep: list = None,
                 units_fc: int = 64, dropout_prob_1: float = 0.0, dropout_prob_2: float = 0.0):
        super(MCV_Net, self).__init__()
        
        # Configurations for the different types of block architectures
        configs = {
            "baseline":             {"use_bn": False, "use_pool": False, "use_gap": False, "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False, "use_prelu": False},
            "maxpool":              {"use_bn": False, "use_pool": True,  "use_gap": False, "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False, "use_prelu": False},
            "maxpool_bn":           {"use_bn": True,  "use_pool": True,  "use_gap": False, "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False, "use_prelu": False},
            "maxpool_gap_bn":       {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False, "use_prelu": False},
            "maxpool_gap_bn_dw":    {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": False, "use_prelu": False},
            "maxpool_gap_bn_dw_at": {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": True,  "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": False, "use_prelu": False},
            "maxpool_gap_bn_dw_r":  {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": True,  "use_dw": True,  "use_conv_pool": False, "use_shuffle": False, "use_prelu": False},
            "maxpool_gap_bn_dw_p":  {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": False, "use_prelu": True},
            "convpool_gap_bn_dw":   {"use_bn": True,  "use_pool": False, "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": True,  "use_conv_pool": True,  "use_shuffle": False, "use_prelu": False},
            "maxpool_gap_bn_dw_sh": {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": True,  "use_prelu": False},
            "maxpool_gap_bn_dw_sh_at":{"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": True, "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": True, "use_prelu": False},
        }
        self.model_cfg = configs[block_type]
        
        # Define blocks using init_chan parameter
        blocks = [MCV_Block(3, init_chan, self.model_cfg, first=True, last=False, dropout_prob_1=dropout_prob_1)]
        if num_blocks > 1:
            mid_blocks=[MCV_Block(init_chan * i, init_chan * (i+1), self.model_cfg, first=False, last=False, dropout_prob_1=dropout_prob_1) for i in range (1, num_blocks-1)]
            blocks.extend(mid_blocks)
            blocks.append(MCV_Block(init_chan * (num_blocks-1), init_chan * num_blocks, self.model_cfg, first=False, last=True, dropout_prob_1=dropout_prob_1))
        
        # Define blocks using filters parameter
        """blocks = [MCV_Block(3, filters[0], self.model_cfg, first=True, last=False)]
        if num_blocks > 1:
            mid_blocks=[MCV_Block(filters[i-1], filters[i], self.model_cfg, first=False, last=False) for i in range (1, num_blocks-1)]
            blocks.extend(mid_blocks)
        blocks.append(MCV_Block(filters[-2], filters[-1], self.model_cfg, first=False, last=True))"""
        
        # Feature extractor backbone
        self.backbone = nn.Sequential(*blocks)
            
        # Calculate number of flatten features before classification head
        if self.model_cfg["use_gap"]: 
            self.gap = nn.AdaptiveAvgPool2d(1) # Creates GAP layer
            flat_features = init_chan * num_blocks # W/ init_chan parameter
            #flat_features=filters[-1] # W/ filters parameter
        else:
            reducer = 2**num_blocks if self.model_cfg["use_pool"] else 1
            flat_features = init_chan * num_blocks * (image_size[0] // reducer) * (image_size[1] // reducer)
        
        # Classification Head
        self.head = ClassificationHead(flat_features, num_classes, self.model_cfg, units_fc, dropout_prob_2=dropout_prob_2)

    def forward(self, x):
        features = self.backbone(x)
        if self.model_cfg["use_gap"]: features = self.gap(features)
        logits = self.head(features)
        
        return logits

    def freeze_backbone(self, freeze):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def ablation(self, blocks_to_keep: list = None):
        if blocks_to_keep is not None:
            all_blocks = list(self.backbone.children())
            self.backbone = nn.Sequential(*[all_blocks[i] for i in blocks_to_keep])
            

    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer: List[Type[nn.Module]], 
                         targets: List[Type[ClassifierOutputTarget]]) -> Type[GradCAMPlusPlus]:

        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam
    
    def prune_backbone(self, sensitivity=0.1):
        """
        Implements element-wise magnitude pruning.
        1. Determines importance based on weight standard deviation.
        2. Stores masks to be used during the training loop.
        """
        self.masks = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Calculate threshold: std deviation * sensitivity
                threshold = torch.std(param.data) * sensitivity
                
                # Create binary mask: 1 if weight > threshold, 0 otherwise
                mask = torch.abs(param.data) > threshold
                self.masks[name] = mask.to(param.device)
                
                # Apply mask immediately to zero out small weights
                param.data *= self.masks[name]
                
        print(f"Pruning complete. Masks generated for {len(self.masks)} layers.")
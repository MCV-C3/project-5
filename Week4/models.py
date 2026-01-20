
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
    def __init__(self, in_f, out_f, model_cfg, first, last):
        super(MCV_Block, self).__init__()
        g = 5 if in_f % 5 == 0 and out_f % 5 == 0 else 4
        
        if model_cfg["use_dw"] and not first:
            # Depthwise convolutions
            layers = [
                nn.Conv2d(in_f, in_f, kernel_size=3, padding=1, stride=1, groups=in_f, bias=not model_cfg["use_bn"]), # Convolutions applied per-channel
                nn.BatchNorm2d(in_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                nn.ReLU(), # Activation
                nn.Conv2d(in_f, out_f, kernel_size=1, groups=g if model_cfg["use_shuffle"] and not last else 1, bias=not model_cfg["use_bn"]), # 1x1 convolution to mix channels
                nn.BatchNorm2d(out_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                ChannelShuffle(groups=g) if model_cfg["use_shuffle"] and not last else nn.Identity(), # Shuffle
                nn.ReLU() # Activation
            ]
        else:
            # Traditional convolution for the first layer and when not using depthwise convolutions
            layers = [
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1, bias=not model_cfg["use_bn"]), # Set bias to False when having a bn layer
                nn.BatchNorm2d(out_f) if model_cfg["use_bn"] else nn.Identity(), # Batch Norm
                nn.ReLU() # Activation
            ]
        
        # Attention layer
        if model_cfg["use_attention"]: layers.append(SpatialAttention())
        #if model_cfg["use_attention"]: layers.append(SEBlock(out_f))
        
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
    def __init__(self, flat_features, num_classes, model_cfg):
        super(ClassificationHead, self).__init__()
        
        if model_cfg["use_gap"]: 
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_features, num_classes)  
            )          
        else:
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_features, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x):
        return self.layers(x)

class MCV_Net(nn.Module):    
    def __init__(self, image_size: int, block_type: str = "baseline", num_classes: int = 8, 
                 init_chan: int = 20, num_blocks: int = 4, filters: list = [12,24,48,96,128]):
        super(MCV_Net, self).__init__()
        
        # Configurations for the different types of block architectures
        configs = {
            "baseline":             {"use_bn": False, "use_pool": False, "use_gap": False, "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False},
            "maxpool":              {"use_bn": False, "use_pool": True,  "use_gap": False, "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False},
            "maxpool_bn":           {"use_bn": True,  "use_pool": True,  "use_gap": False, "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False},
            "maxpool_gap_bn":       {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": False, "use_conv_pool": False, "use_shuffle": False},
            "maxpool_gap_bn_dw":    {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": False},
            "maxpool_gap_bn_dw_at": {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": True,  "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": False},
            "maxpool_gap_bn_dw_r":  {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": True,  "use_dw": True,  "use_conv_pool": False, "use_shuffle": False},
            "convpool_gap_bn_dw":   {"use_bn": True,  "use_pool": False, "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": True,  "use_conv_pool": True,  "use_shuffle": False},
            "maxpool_gap_bn_dw_sh": {"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": False, "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": True},
            "maxpool_gap_bn_dw_sh_at":{"use_bn": True,  "use_pool": True,  "use_gap": True,  "use_attention": True, "use_residual": False, "use_dw": True,  "use_conv_pool": False, "use_shuffle": True},
        }
        self.model_cfg = configs[block_type]
        
        # Define blocks using init_chan parameter
        blocks = [MCV_Block(3, init_chan, self.model_cfg, first=True, last=False)]
        if num_blocks > 1:
            mid_blocks=[MCV_Block(init_chan * i, init_chan * (i+1), self.model_cfg, first=False, last=False) for i in range (1, num_blocks-1)]
            blocks.extend(mid_blocks)
            blocks.append(MCV_Block(init_chan * (num_blocks-1), init_chan * num_blocks, self.model_cfg, first=False, last=True))
        
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
        self.head = ClassificationHead(flat_features, num_classes, self.model_cfg)

    def forward(self, x):
        features = self.backbone(x)
        if self.model_cfg["use_gap"]: features = self.gap(features)
        logits = self.head(features)
        
        return logits

    def extract_feature_maps_by_block(self, input_image: torch.Tensor):
        """
        Extracts output from the first 16 main blocks of the backbone.
        """
        feature_maps = []
        layer_names = []
        hooks = []

        def hook_fn(name):
            def rel_hook(module, input, output):
                # output.detach() is critical to avoid memory leaks
                feature_maps.append(output.detach())
                layer_names.append(name)
            return rel_hook

        # Target direct children (Block 0 to 15)
        for i, (name, module) in enumerate(self.backbone.features.named_children()):
            if i < 16:
                hooks.append(module.register_forward_hook(hook_fn(f"Block_{i}")))

        with torch.no_grad():
            self.backbone(input_image)

        # Always remove hooks to keep the model clean
        for hook in hooks:
            hook.remove()

        return feature_maps, layer_names

    def extract_features_from_hooks(self, x, layers: List[str]):
        """
        Extract feature maps from specified layers.
        Args:
            x (torch.Tensor): Input tensor.
            layers (List[str]): List of layer names to extract features from.
        Returns:
            Dict[str, torch.Tensor]: Feature maps from the specified layers.
        """
        outputs = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                outputs[name] = output
            return hook

        # Register hooks for specified layers
        #for layer_name in layers:
        dict_named_children = {}
        for name, layer in self.backbone.named_children():
            for n, specific_layer in layer.named_children():
                dict_named_children[f"{name}.{n}"] = specific_layer

        for layer_name in layers:
            layer = dict_named_children[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs

    def modify_layers(self, modify_fn: Callable[[nn.Module], nn.Module]):
        """
        Modify layers of the model using a provided function.
        Args:
            modify_fn (Callable[[nn.Module], nn.Module]): Function to modify a layer.
        """
        self.mobileNet = modify_fn(self.mobileNet)


    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False



    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer: List[Type[nn.Module]], 
                         targets: List[Type[ClassifierOutputTarget]]) -> Type[GradCAMPlusPlus]:

        

        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:

            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam





# Example of usage
if __name__ == "__main__":
    IMG_SIZE=(224, 224)
    
    torch.manual_seed(42)

    # Load a pretrained model and modify it
    model = WraperModel(num_classes=8, feature_extraction=False)
    model.load_state_dict(torch.load("./saved_models/fold-1_0.001_20.pt"))
    model.eval()
    
    idx_to_label = {
        0: "Opencountry",
        1: "coast",
        2: "forest",
        3: "highway",
        4: "inside_city",
        5: "mountain",
        6: "street",
        7: "tallbuilding"
    }
    
    """print(f"Total capas en features: {len(model.backbone.features)}")
    for i, layer in enumerate(model.backbone.features):
        print(f"Index {i}: {layer}")"""

    transformation  = T.Compose([
                                    T.ToImage(),
                                    T.ToDtype(torch.float32, scale=True),
                                    T.Resize(size=IMG_SIZE),
                                ])
    # Example GradCAM usage
    path = os.path.expanduser("~/mcv/datasets/C3/2425/MIT_small_train_1/test/street/bost46.jpg")
    dummy_input = Image.open(path)
    input_image = transformation(dummy_input).unsqueeze(0)

    target_layers = [model.backbone.features[16]] # Compute Grad Cam on last feature extraction layer
    #target_layers = [model.backbone.features[15].block[-1]]
    targets = [ClassifierOutputTarget(0)] # Number of the class that is being looked
    
    with torch.no_grad():
        output = model(input_image)
        _, predicted = output.max(1)
    print(f"Output for input image: {output}")
    print(f"Prediction for input image: {idx_to_label.get(predicted.item())}")
    
    img_for_vis = np.array(dummy_input.resize(IMG_SIZE)).astype(np.float32) / 255.0

    ## Visualize the activation map from Grad Cam
    ## To visualize this, it is mandatory to have gradients.
    grad_cams = model.extract_grad_cam(input_image=input_image, target_layer=target_layers, targets=targets)
    visualization = show_cam_on_image(img_for_vis, grad_cams, use_rgb=True)

    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.savefig("./figures/grad_cam.png")
    plt.close("all")

    # 1. Run extraction
    f_maps, l_names = model.extract_feature_maps_by_block(input_image)

    # 2. Setup 4x4 Grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(16):
        # Process: Squeeze batch -> Mean across channels -> CPU
        # Mean helps visualize the general activation area
        f_map = torch.mean(f_maps[i].squeeze(0), dim=0).cpu()
        
        # Min-Max Normalization: Forces contrast to avoid flat colors
        f_min, f_max = f_map.min(), f_map.max()
        if f_max > f_min:
            f_map = (f_map - f_min) / (f_max - f_min)
        else:
            f_map = torch.zeros_like(f_map)
        
        axes[i].imshow(f_map.numpy(), cmap='viridis')
        axes[i].set_title(f"{l_names[i]} (Res: {f_map.shape[0]}x{f_map.shape[1]})", fontsize=16)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("./figures/feature_maps.png", dpi=150)
    plt.close(fig)

    ## Plot a concret layer feature map when processing a image thorugh the model
    ## Is not necessary to have gradients

    """with torch.no_grad():
        feature_map = (model.extract_features_from_hooks(x=input_image, layers=["features.28"]))["features.28"]
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        print(feature_map.shape)
        processed_feature_map, _ = torch.min(feature_map, 0) 

    # Plot the result
    plt.imshow(processed_feature_map, cmap="gray")
    plt.axis("off")
    plt.show()"""



    ## Draw the model
    #model_graph = draw_graph(model, input_size=(1, 3, 224, 224), device='meta', expand_nested=True, roll=True)
    #model_graph.visual_graph.render(filename="test", format="png", directory="./Week3")
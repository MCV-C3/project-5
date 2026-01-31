
import torch.nn as nn
import torch
import copy
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import models
from transformers import ViTModel, SwinModel

import matplotlib.pyplot as plt

from typing import *
from torchview import draw_graph
from graphviz import Source

from PIL import Image
import torchvision.transforms.v2  as F
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
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True

class WraperModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_extraction: bool = True,
        blocks_to_keep: Optional[List[int]] = None,
        out_feat: int = -1,
        dropout_prob: float = 0.2,
        backbone_name: str = "mobilenet"
    ):

        super(WraperModel, self).__init__()
        """
        # Load pretrained MobileNet model
        self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        
        if feature_extraction:
            self.set_parameter_requires_grad(feature_extracting=feature_extraction)
        
        # Modify the internal features blocks keeping "blocks_to_kep"
        if blocks_to_keep is not None:
            # We extract the original blocks and reconstruct the features
            all_blocks = list(self.backbone.features.children())
            self.backbone.features = nn.Sequential(*[all_blocks[i] for i in blocks_to_keep])
            
            # Recalculate the classifier input because removing blocks changes the depth
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                new_in_channels = self.backbone.features(dummy_input).shape[1]
                if out_feat == -1: out_features = self.backbone.classifier[0].out_features
                else: out_features = out_feat
            
            # Modify the classifier head
            self.backbone.classifier = nn.Sequential(
                nn.Linear(new_in_channels, out_features),
                nn.Hardswish(),
                nn.Dropout(p=dropout_prob, inplace=True),
                nn.Linear(out_features, num_classes)
            )
            
        else:
            # Modify the classifier for the number of classes
            self.backbone.classifier[2] = nn.Dropout(p=dropout_prob, inplace=True)
            self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)
        """
        self.backbone_name = backbone_name

        # ==========================
        # MobileNet backbone
        # ==========================
        if backbone_name == "mobilenet":
            self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

            if feature_extraction:
                self.set_parameter_requires_grad(feature_extracting=True)

            # ----- BLOCK PRUNING (MobileNet only) -----
            if blocks_to_keep is not None:
                all_blocks = list(self.backbone.features.children())
                self.backbone.features = nn.Sequential(*[all_blocks[i] for i in blocks_to_keep])

                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    new_in_channels = self.backbone.features(dummy_input).shape[1]
                    out_features = (
                        self.backbone.classifier[0].out_features
                        if out_feat == -1 else out_feat
                    )

                self.backbone.classifier = nn.Sequential(
                    nn.Linear(new_in_channels, out_features),
                    nn.Hardswish(),
                    nn.Dropout(p=dropout_prob, inplace=True),
                    nn.Linear(out_features, num_classes)
                )
            else:
                self.backbone.classifier[2] = nn.Dropout(p=dropout_prob, inplace=True)
                self.backbone.classifier[-1] = nn.Linear(
                    self.backbone.classifier[-1].in_features,
                    num_classes
                )

        # ==========================
        # ViT backbone
        # ==========================
        elif backbone_name == "vit":
            self.backbone_name = "vit"

            # ---- Hugging Face ViT backbone ----
            self.backbone = ViTModel.from_pretrained(
                "google/vit-base-patch16-224"
            )

            hidden_dim = self.backbone.config.hidden_size

            # Classification head (CLS token)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, num_classes)
            )

            # Freeze backbone if feature extraction
            if feature_extraction:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            if blocks_to_keep is not None:
                raise NotImplementedError("blocks_to_keep is not supported for ViT")
        # ==================================================
        # Swin (Hugging Face)
        # ==================================================
        elif backbone_name == "swin":
            self.backbone = SwinModel.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224"
            )

            hidden_dim = self.backbone.config.hidden_size

            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, num_classes)
            )

            if feature_extraction:
                for p in self.backbone.parameters():
                    p.requires_grad = False

            if blocks_to_keep is not None:
                raise NotImplementedError("blocks_to_keep not supported for Swin")
                
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")


    def forward(self, x):
        if self.backbone_name == "vit":
            outputs = self.backbone(pixel_values=x)
            cls_token = outputs.last_hidden_state[:, 0]  # [B, hidden]
            return self.classifier(cls_token)
        elif self.backbone_name == "swin":
            outputs = self.backbone(pixel_values=x)
            pooled = outputs.pooler_output
            return self.classifier(pooled)
        else:
            return self.backbone(x)

    

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

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=IMG_SIZE),
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
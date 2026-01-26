import torch
import matplotlib as plt
from PIL import Image
import os
import torchvision.transforms.v2  as F
import numpy as np 
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import MCV_Net

if __name__ == "__main__":
    IMG_SIZE=(224, 224)
    
    # UNIQUE PARAMETERS TO SET
    # True if you want Grad-CAM calculated from gt label. If desired Grad-CAM from predicted label set to False
    GT_GRAD_CAM = True
    CLASS_IDX = 1
    BLOCK_TYPE="maxpool_gap_bn" # Model with GAP
    #BLOCK_TYPE="maxpool_bn" # Model without GAP
    
    idx_to_label = {0: "Opencountry", 1: "coast", 2: "forest", 3: "highway", 
                    4: "inside_city", 5: "mountain", 6: "street", 7: "tallbuilding"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(42)

    model = MCV_Net(image_size=(224, 224), block_type=BLOCK_TYPE, num_classes=8, num_blocks=4, init_chan=20)
    model.load_state_dict(torch.load(f"./saved_models/fold_1_{BLOCK_TYPE}.pt")) # Charge the model
    model.to(device)
    model.eval()

    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=IMG_SIZE),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Search for a False Negative in a given folder ---
    folder_path = os.path.expanduser(f"~/mcv/datasets/C3/2425/MIT_small_train_1/test/{idx_to_label[CLASS_IDX]}/")
    found_fn = False

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        dummy_input = Image.open(path).convert('RGB')
        input_image = transformation(dummy_input).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_image)
            _, predicted = output.max(1)
            pred_idx = predicted.item()

        if pred_idx != CLASS_IDX:
            print(f"False Negative found: {filename} predicted as {idx_to_label[pred_idx]}")
            found_fn = True
            break # Stop loop and continue with this image

    if not found_fn:
        print("No False Negatives found in the folder.")
        exit()

    # --- GradCAM Logic ---
    target_layers = [model.backbone[3].block[2]] # Grad-CAM computed from last layer of feature extractor
    # Explaining the WRONG prediction found
    targets = [ClassifierOutputTarget(CLASS_IDX if GT_GRAD_CAM else pred_idx)]
    
    # Required for GradCAM
    input_image.requires_grad = True
    
    img_for_vis = np.array(dummy_input.resize(IMG_SIZE)).astype(np.float32) / 255.0

    grad_cams = model.extract_grad_cam(input_image=input_image, target_layer=target_layers, targets=targets)
    visualization = show_cam_on_image(img_for_vis, grad_cams, use_rgb=True)

    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.title(f"GT: {idx_to_label[CLASS_IDX]} | Pred: {idx_to_label[pred_idx]}")
    plt.savefig("./figures/grad_cam.png")
    plt.close("all")

    # --- Feature Maps Logic ---
    """f_maps, l_names = model.extract_feature_maps_by_block(input_image)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(16):
        f_map = torch.mean(f_maps[i].squeeze(0), dim=0).detach().cpu()
        
        f_min, f_max = f_map.min(), f_map.max()
        if f_max > f_min:
            f_map = (f_map - f_min) / (f_max - f_min)
        else:
            f_map = torch.zeros_like(f_map)
        
        axes[i].imshow(f_map.numpy(), cmap='viridis')
        axes[i].set_title(f"{l_names[i]} (Res: {f_map.shape[0]}x{f_map.shape[1]})", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("./figures/feature_maps.png", dpi=150)
    plt.close(fig)"""
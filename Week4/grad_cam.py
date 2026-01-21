import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import torchvision.transforms.v2 as F
import numpy as np 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Assuming models.py is in the same directory
from models import MCV_Net

# --- Helper Classes ---
class FlattenModelWrapper(torch.nn.Module):
    """
    Wraps the model to ensure the output is flattened (batch_size, num_classes)
    before being passed to GradCAM, preventing RuntimeError on non-scalar outputs.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x).flatten(1)

# --- Core Functions ---

def load_model(block_type, num_classes, device, weights_path):
    """Initializes and loads the model weights."""
    print(f"Loading model: {block_type} from {weights_path}...")
    model = MCV_Net(image_size=(224, 224), block_type=block_type, num_classes=num_classes, num_blocks=4, init_chan=20)
    
    # Use weights_only=True to avoid FutureWarning and security risks
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_transform(img_size=(224, 224)):
    """Returns the validation/test preprocessing pipeline."""
    return F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=img_size),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_image(image_path, transformation, device):
    """Opens an image and applies transformations."""
    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = transformation(raw_image).unsqueeze(0).to(device)
    return raw_image, input_tensor

def predict(model, input_tensor):
    """Returns the predicted class index."""
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
    return predicted.item()

def find_random_false_negative(model, folder_path, true_label_idx, transformation, device, idx_to_label):
    """
    Shuffles the folder contents and returns the first False Negative found.
    """
    print(f"Searching for a random False Negative (GT: {idx_to_label[true_label_idx]})...")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return None, None, None, None

    # Get list of files
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # SHUFFLE to find different images each run
    random.shuffle(files)

    for filename in files:
        path = os.path.join(folder_path, filename)
        try:
            raw_img, input_tensor = process_image(path, transformation, device)
            pred_idx = predict(model, input_tensor)

            if pred_idx != true_label_idx:
                print(f"False Negative found: {filename} | Predicted: {idx_to_label.get(pred_idx, pred_idx)}")
                return raw_img, input_tensor, pred_idx, filename
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            continue

    print("No False Negatives found in the folder.")
    return None, None, None, None

def compute_gradcam(model, input_tensor, raw_image, target_layers, vis_class_id=None):
    """
    Computes Grad-CAM.
    vis_class_id: The specific class index to visualize. 
                  If None, visualizes the class with the highest confidence score.
    """
    input_tensor.requires_grad = True
    
    # Prepare image for visualization (H, W, 3) normalized 0-1
    img_size = (input_tensor.shape[2], input_tensor.shape[3])
    img_for_vis = np.array(raw_image.resize(img_size)).astype(np.float32) / 255.0

    # Define targets
    if vis_class_id is None:
        targets = None # Auto-selects highest confidence
    else:
        targets = [ClassifierOutputTarget(vis_class_id)]

    # Initialize GradCAM with the wrapped model
    cam_algorithm = GradCAM(model=FlattenModelWrapper(model), target_layers=target_layers)

    # Generate CAM
    # We select [0, :] because batch size is 1
    grad_cams = cam_algorithm(input_tensor=input_tensor, targets=targets)[0, :]
    
    visualization = show_cam_on_image(img_for_vis, grad_cams, use_rgb=True)
    return visualization

def visualize_result(visualization, gt_label, pred_label, vis_label, filename, save_path="./figures/grad_cam.png"):
    """Plots and saves the result."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.axis("off")
    plt.title(f"File: {filename}\nGT: {gt_label} | Pred: {pred_label} | Vis Neuron: {vis_label}")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    plt.close("all")


if __name__ == "__main__":
    # ================= CONFIGURATION =================
    BLOCK_TYPE = "maxpool_bn"
    BLOCK_TYPE = "maxpool_gap_bn"
    
    NUM_CLASSES = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    
    # Dataset / Labels
    IDX_TO_LABEL = {0: "Opencountry", 1: "coast", 2: "forest", 3: "highway", 
                    4: "inside_city", 5: "mountain", 6: "street", 7: "tallbuilding"}
    
    # Search Settings
    SEARCH_CLASS_IDX = 5  # The Ground Truth label to look for (e.g., 'coast')
    DATASET_ROOT = os.path.expanduser(f"~/mcv/datasets/C3/2425/MIT_small_train_1/test/{IDX_TO_LABEL[SEARCH_CLASS_IDX]}/")
    
    # Single Image Mode (Set path to process 1 specific image, or None to search folder)
    SINGLE_IMAGE_PATH = f"/ghome/group05/mcv/datasets/C3/2425/MIT_small_train_1/test/{IDX_TO_LABEL[SEARCH_CLASS_IDX]}/nat11.jpg" 
    SINGLE_IMAGE_PATH = None 

    # Neuron Selection
    # If None: Visualize neuron with highest confidence (Predicted Class)
    # If Integer: Visualize specific class (e.g., 1 for coast)
    VIS_CLASS_ID = 1
    # =================================================

    # 1. Load Model
    weights_file = f"./saved_models/fold_1_{BLOCK_TYPE}.pt"
    model = load_model(BLOCK_TYPE, NUM_CLASSES, DEVICE, weights_file)
    
    # Define Target Layer for GradCAM (Last layer of feature extractor)
    target_layers = [model.backbone[3].block[2]] 

    # 2. Get Input Image
    transformation = get_transform()
    
    raw_img, input_tensor, pred_idx, filename = None, None, None, None

    if SINGLE_IMAGE_PATH:
        print(f"Processing single image: {SINGLE_IMAGE_PATH}")
        raw_img, input_tensor = process_image(SINGLE_IMAGE_PATH, transformation, DEVICE)
        pred_idx = predict(model, input_tensor)
        filename = os.path.basename(SINGLE_IMAGE_PATH)
    else:
        # Search for Random False Negative
        raw_img, input_tensor, pred_idx, filename = find_random_false_negative(
            model, DATASET_ROOT, SEARCH_CLASS_IDX, transformation, DEVICE, IDX_TO_LABEL
        )

    if raw_img is None:
        print("No image found to process. Exiting.")
        exit()

    # 3. Compute GradCAM
    # Determine label names for title
    gt_label_name = IDX_TO_LABEL.get(SEARCH_CLASS_IDX, "Unknown")
    pred_label_name = IDX_TO_LABEL.get(pred_idx, "Unknown")
    
    # If VIS_CLASS_ID is None, we are visualizing the predicted class
    vis_id_used = pred_idx if VIS_CLASS_ID is None else VIS_CLASS_ID
    vis_label_name = IDX_TO_LABEL.get(vis_id_used, str(vis_id_used))
    if VIS_CLASS_ID is None:
        vis_label_name += " (Max Conf)"

    print(f"Computing GradCAM for class: {vis_label_name}...")
    
    cam_vis = compute_gradcam(model, input_tensor, raw_img, target_layers, vis_class_id=VIS_CLASS_ID)

    # 4. Visualize
    save_path = f"./figures/grad_cam_{IDX_TO_LABEL[SEARCH_CLASS_IDX]}_neuron_{IDX_TO_LABEL.get(vis_id_used, str(vis_id_used))}_{'GAP' if 'gap' in BLOCK_TYPE else 'noGAP'}.png" 
    visualize_result(cam_vis, gt_label_name, pred_label_name, vis_label_name, filename, save_path)
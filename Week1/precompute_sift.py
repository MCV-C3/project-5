import os
import glob
import numpy as np
import cv2
import tqdm
from PIL import Image

# Import your existing class
from bovw import BOVW

DATA_ROOT = "/home/bernat/MCV/C3/project/project-5/data/places_reduced/"
SPLITS = ["train", "val"]  # Folders to process
DESCRIPTOR_TYPE = "DENSE_SIFT"
STEP_SIZE = 8 # Important for Dense SIFT grid density
SAVE_EXTENSION = ".npy" # NumPy binary format

def precompute_dataset():
    bovw = BOVW(detector_type=DESCRIPTOR_TYPE)
    
    print(f"Starting Precomputation for {DESCRIPTOR_TYPE}...")

    for split in SPLITS:
        split_path = os.path.join(DATA_ROOT, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split folder not found: {split_path}")
            continue

        # Get all class folders (e.g., 'kitchen', 'bedroom')
        class_folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
        
        for cls in tqdm.tqdm(class_folders, desc=f"Processing {split}"):
            class_path = os.path.join(split_path, cls)
            
            # Create the output folder INSIDE the class folder
            output_dir = os.path.join(class_path, "precomputed_sift")
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all images (jpg)
            image_files = glob.glob(os.path.join(class_path, "*.jpg"))
            
            for img_path in image_files:
                try:
                    img_pil = Image.open(img_path).convert("RGB")
                    img_np = np.array(img_pil)
                    
                    # Extract Features
                    _, descriptors = bovw._extract_features(
                        image=img_np, 
                        step_size=STEP_SIZE
                    )
                    
                    if descriptors is None:
                        descriptors = np.zeros((0, 128), dtype=np.float32)

                    # Save to disk
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_path = os.path.join(output_dir, base_name + SAVE_EXTENSION)
                    
                    np.save(save_path, descriptors)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print("\nPrecomputation Complete!")
    print(f"Descriptors saved in: .../<class_label>/precomputed_sift/*.npy")

if __name__ == "__main__":
    precompute_dataset()
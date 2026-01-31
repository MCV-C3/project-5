import os
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F
from PIL import Image
from tqdm import tqdm
import math

def save_patches(input_dir, output_dir, patches_lvl):
    """
    Reads images from input_dir, crops them into grids, and saves patches to output_dir.
    """
    # 1. Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    grid_size = 2 ** patches_lvl
    print(f"Generating Level {patches_lvl} patches (Grid: {grid_size}x{grid_size})...")

    # We use ImageFolder just to get the list of file paths and classes easily
    dataset = ImageFolder(input_dir)
    
    # 2. Iterate through all images
    for idx, (img_path, target) in enumerate(tqdm(dataset.samples)):
        # img_path is the full path, target is the class index
        
        # Get class name to maintain folder structure
        class_name = dataset.classes[target]
        file_name = os.path.basename(img_path)
        base_name, ext = os.path.splitext(file_name)
        
        # Create class directory in output
        save_path_class = os.path.join(output_dir, class_name)
        os.makedirs(save_path_class, exist_ok=True)
        
        # Load Image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            
            # Calculate Patch Size
            step_w, step_h = w // grid_size, h // grid_size
            
            # 3. Crop and Save
            count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    top = i * step_h
                    left = j * step_w
                    
                    # Crop: (left, top, right, bottom)
                    patch = img.crop((left, top, left + step_w, top + step_h))
                    
                    # Construct new filename: imageID_row_col.jpg
                    patch_filename = f"{base_name}_p{i}_{j}{ext}"
                    patch.save(os.path.join(save_path_class, patch_filename))
                    count += 1

if __name__ == "__main__":
    images_dir = "/ghome/group05/mcv_patches/datasets/C3/2526/places_reduced/"
    patches_lvl = 1
    train = os.path.join(images_dir, "train")
    val = os.path.join(images_dir, "val")
    classes = os.listdir(train)
    
    for clas in classes:
        print(f"Processing class: {clas}")
        input_dir = os.path.join(train, clas)
        output_dir = os.path.join(input_dir, f"patches_lvl_{patches_lvl}/")
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        # output_dir = os.path.join("mcv/datasets/C3/2526/places_patches/train", clas)
        
    
        save_patches(input_dir, output_dir, patches_lvl)
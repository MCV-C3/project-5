import os
import matplotlib.pyplot as plt
import numpy as np
import glob

BASE_PATH = "/ghome/group05/mcv/datasets/C3/2425/"

FOLDS_TO_ANALYZE = [
    "MIT_large_train",
    "MIT_small_train_1",
    "MIT_small_train_2",
    "MIT_small_train_3",
    "MIT_small_train_4"
]

IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')

def get_class_counts():
    fold_data = {}
    first_fold_path = os.path.join(BASE_PATH, FOLDS_TO_ANALYZE[0], 'train')
    if not os.path.exists(first_fold_path):
        print(f"Error: Path not found {first_fold_path}")
        return None, None
        
    classes = sorted([d for d in os.listdir(first_fold_path) if os.path.isdir(os.path.join(first_fold_path, d))])
    print(f"Classes detected ({len(classes)}): {classes}")

    for fold_name in FOLDS_TO_ANALYZE:
        print(f"Processing {fold_name}...")
        fold_data[fold_name] = {'train': [], 'test': []}
        
        for split in ['train', 'test']:
            counts = []
            for cls in classes:
                path = os.path.join(BASE_PATH, fold_name, split, cls)
                n_files = 0
                for ext in IMAGE_EXTENSIONS:
                    n_files += len(glob.glob(os.path.join(path, ext)))
                    n_files += len(glob.glob(os.path.join(path, ext.upper())))
                counts.append(n_files)
            fold_data[fold_name][split] = counts
            
    return fold_data, classes

def plot_analysis(data, classes):
    figures = []
    
    x = np.arange(len(classes))
    width = 0.35
    
    c_train = '#2980b9'
    c_test = '#c0392b'

    for fold_name in FOLDS_TO_ANALYZE:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        train_counts = data[fold_name]['train']
        test_counts = data[fold_name]['test']
        
        max_count = max(max(train_counts) if train_counts else 0, max(test_counts) if test_counts else 0)
        y_limit = max_count * 1.15 if max_count > 0 else 10
        
        ax1.bar(x - width/2, train_counts, width, label='Train', color=c_train, alpha=0.85, edgecolor='black', linewidth=0.7)
        ax1.set_ylabel('Train Count', color=c_train, fontweight='bold', fontsize=13)
        ax1.tick_params(axis='y', labelcolor=c_train, labelsize=11)
        ax1.set_ylim(0, y_limit)
        
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, test_counts, width, label='Test', color=c_test, alpha=0.85, edgecolor='black', linewidth=0.7)
        ax2.set_ylabel('Test Count', color=c_test, fontweight='bold', fontsize=13)
        ax2.tick_params(axis='y', labelcolor=c_test, labelsize=11)
        ax2.set_ylim(0, y_limit)
        
        ax1.set_title(f'Class Distribution - {fold_name}', fontweight='bold', fontsize=16, pad=15)
        ax1.set_xlabel('Classes', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=12, fontweight='medium')
        ax1.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=12, framealpha=0.95)
        
        plt.tight_layout()
        figures.append((fold_name, fig))
    
    return figures


if __name__ == "__main__":
    if not os.path.exists(BASE_PATH):
        print(f"Error: Path not found: {BASE_PATH}")
    else:
        fold_data, class_names = get_class_counts()
        if fold_data:
            # Create output directory
            output_dir = 'res_folds_analysis'
            os.makedirs(output_dir, exist_ok=True)
            
            figures = plot_analysis(fold_data, class_names)
            for fold_name, fig in figures:
                output_file = os.path.join(output_dir, f'analysis_{fold_name}.png')
                fig.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
                print(f"Saved: {output_file}")
                plt.close(fig)
            print("\nAll plots generated successfully.")
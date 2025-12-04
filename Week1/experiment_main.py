import os
import glob
import numpy as np
import tqdm
import pickle
import wandb
import itertools
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing classes
from bovw import BOVW

class PrecomputedDataset:
    """
    A specific dataset loader that looks for .npy files 
    inside the 'precomputed_sift' subfolders.
    """
    def __init__(self, root_folder: str, split: str = "train"):
        self.data: List[Tuple[str, int]] = []
        self.classes = {}
        
        split_path = os.path.join(root_folder, split)
        
        # Map class names to integers
        # Expected: root/train/<class>/precomputed_sift/*.npy
        class_names = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        class_names.sort()
        self.classes = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # Collect all .npy file paths
        for cls_name in class_names:
            # Look specifically in the 'precomputed_sift' folder created previously
            precomp_dir = os.path.join(split_path, cls_name, "precomputed_sift")
            
            if not os.path.exists(precomp_dir):
                print(f"Warning: No precomputed folder for class '{cls_name}'")
                continue
                
            files = glob.glob(os.path.join(precomp_dir, "*.npy"))
            for f in files:
                self.data.append((f, self.classes[cls_name]))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        # Load the descriptors directly
        descriptors = np.load(path)
        return descriptors, label

def batch_generator(dataset, batch_size):
    """Yields batches of (descriptors, labels) from the dataset."""
    for i in range(0, len(dataset), batch_size):
        # Slice the dataset list
        batch_items = dataset.data[i:i + batch_size]
        
        # Load the actual data for this batch
        loaded_batch = []
        for path, label in batch_items:
            descriptors = np.load(path)
            loaded_batch.append((descriptors, label))
            
        yield loaded_batch

class ExperimentRunner:
    def __init__(self, config: Dict[str, Any], project_name: str = "bovw-project"):
        self.config = config
        self.project_name = project_name
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.run_name = f"Precomp_Exp_{config['detector_type']}_{config['codebook_size']}k_{timestamp}"
        
        self.bovw = None
        self.classifier = None
        self.train_data = None
        self.val_data = None

    def setup_wandb(self):
        wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            reinit=True
        )

    def load_data(self):
        print(f"[{self.run_name}] Indexing Precomputed Files...")
        self.train_data = PrecomputedDataset(self.config['data_path'], split="train")
        self.val_data = PrecomputedDataset(self.config['data_path'], split="val")
        print(f"Found {len(self.train_data)} training and {len(self.val_data)} validation files.")

    def run_pipeline_batched(self):
        # Initialize BOVW
        # We don't strictly need detector_kwargs since extraction is already done,
        # but we need the class to handle K-Means logic.
        self.bovw = BOVW(
            detector_type=self.config['detector_type'],
            codebook_size=self.config['codebook_size'],
            codebook_kwargs=self.config.get('codebook_kwargs', {})
        )

        # --- Phase 1: Train Codebook (Batched) ---
        print(f"[{self.run_name}] Phase 1: Learning Codebook (Batched)...")
        
        loader = batch_generator(self.train_data, self.config['batch_size'])
        total_batches = (len(self.train_data) + self.config['batch_size'] - 1) // self.config['batch_size']
        
        for batch in tqdm.tqdm(loader, total=total_batches, desc="Fitting K-Means"):
            # Stack all descriptors in the batch into one big matrix
            # Check if descriptor array is not empty (it might be 0x128)
            valid_descriptors = [d for d, _ in batch if d.shape[0] > 0]
            
            if valid_descriptors:
                # _update_fit_codebook uses partial_fit internally
                self.bovw._update_fit_codebook(descriptors=valid_descriptors)

        print("Codebook fitting complete.")

        # --- Phase 2: Compute Histograms & Classifier ---
        print(f"[{self.run_name}] Phase 2: Computing Histograms & Training Classifier...")
        
        train_hists = []
        train_labels = []

        # Re-iterate dataset to encode
        loader = batch_generator(self.train_data, self.config['batch_size'])
        
        for batch in tqdm.tqdm(loader, total=total_batches, desc="Encoding Training Set"):
            for descriptors, label in batch:
                if descriptors.shape[0] > 0:
                    # Predict visual words and create histogram
                    hist = self.bovw._compute_codebook_descriptor(descriptors, self.bovw.codebook_algo)
                    train_hists.append(hist)
                    train_labels.append(label)
                else:
                    # Return zero histogram
                    train_hists.append(np.zeros(self.config['codebook_size']))
                    train_labels.append(label)

        # Fit Classifier
        print(f"[{self.run_name}] Phase 3: Fitting Logistic Regression...")
        self.classifier = LogisticRegression(
            C=self.config['C_param'], 
            class_weight="balanced", 
            max_iter=100
        )
        self.classifier.fit(train_hists, train_labels)

        # Log Training Accuracy
        train_pred = self.classifier.predict(train_hists)
        train_acc = accuracy_score(train_labels, train_pred)
        wandb.log({"train_accuracy": train_acc})
        print(f"Train Accuracy: {train_acc:.4f}")

    def evaluate(self):
        print(f"[{self.run_name}] Evaluating on Validation Set...")
        val_hists = []
        val_labels = []
        
        # Load validation data
        for i in tqdm.tqdm(range(len(self.val_data)), desc="Encoding Val Set"):
            descriptors, label = self.val_data[i]
            
            if descriptors.shape[0] > 0:
                hist = self.bovw._compute_codebook_descriptor(descriptors, self.bovw.codebook_algo)
            else:
                hist = np.zeros(self.config['codebook_size'])
                
            val_hists.append(hist)
            val_labels.append(label)

        y_pred = self.classifier.predict(val_hists)
        acc = accuracy_score(val_labels, y_pred)
        
        wandb.log({"val_accuracy": acc})
        
        # Confmat
        cm = confusion_matrix(val_labels, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.run_name}')
        wandb.log({"confusion_matrix_plot": wandb.Image(plt)})
        plt.close()

        print(f"Validation Accuracy: {acc:.4f}")

    def save_artifacts(self):
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{self.run_name}_model.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump({'bovw': self.bovw, 'classifier': self.classifier}, f)
        
        artifact = wandb.Artifact(f"model-{self.run_name}", type='model')
        artifact.add_file(filename)
        wandb.log_artifact(artifact)
        print(f"Saved artifacts to {filename}")

    def run(self):
        self.setup_wandb()
        self.load_data()
        self.run_pipeline_batched()
        self.evaluate()
        self.save_artifacts()
        wandb.finish()

if __name__ == "__main__":
    
    # 
    
    # 1. Define Path to the Root (containing train/val folders)
    DATA_PATH = "/home/bernat/MCV/C3/project/project-5/data/places_reduced/"

    # 2. Hyperparameters
    # Note: 'detector_type' is just for naming here, as data is already precomputed.
    # Ensure 'detector_type' matches what you used to generate the .npy files!
    param_grid = {
        'data_path': [DATA_PATH],
        'detector_type': ['DENSE_SIFT'], 
        'codebook_size': [1000],
        'batch_size': [2048],               # Number of FILES to load at once
        'C_param': [1.0],
        'codebook_kwargs': [{'batch_size': 1024*4, 'random_state': 42}] # MiniBatchKMeans specific batch size
    }

    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total experiments to run: {len(experiments)}")

    for i, config in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        runner = ExperimentRunner(config, project_name="Places-BoVW-Precomputed")
        runner.run()
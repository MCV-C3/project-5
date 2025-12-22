from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2  as F
import tqdm
import wandb
from sklearn.utils.class_weight import compute_class_weight
import collections

from models import WraperModel, EarlyStopping
from main import train, test
from metrics import FoldMetrics

import os

BASE_PATH = "~/mcv/datasets/C3/2425/"

def get_loaders_for_fold(fold_idx, batch_size=32, transform_train=None, transform_test=None, num_workers=8):
    fold_dir = f"MIT_small_train_{fold_idx}"
    
    train_path = os.path.join(BASE_PATH, fold_dir, "train")
    test_path = os.path.join(BASE_PATH, fold_dir, "test")
    
    train_dataset = ImageFolder(train_path, transform=transform_train)
    test_dataset = ImageFolder(test_path, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def run_experiment(wandb_config=None, experiment_config=None):
    # Default configurations
    default_wandb_config = {
        "project": "C3-Week3-MobileNet",
        "entity": "project-5",
    }
    
    default_experiment_config = {
        "image_size": (224, 224),
        "batch_size": 256,
        "num_epochs": 20,
        "learning_rate": 0.001,
        "output_dim": 11,
        "num_workers": 8,
        "patience": 5,
        "min_delta": 0,
        "save_weights": False,
        "k_folds": 5,
    }
    
    # Merge configs
    if wandb_config is None: wandb_config = default_wandb_config
    else: wandb_config = {**default_wandb_config, **wandb_config}
    
    if experiment_config is None: experiment_config = default_experiment_config
    else: experiment_config = {**default_experiment_config, **experiment_config}

    # Init wandb
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        config=experiment_config,
    )

    cfg = wandb.config
    print(f"Starting experiment with config: {cfg}")
    
    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=cfg.image_size),
                                ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- HISTORY STORAGE ---
    # history[epoch]['val_acc'] = [fold1, fold2...]
    history = collections.defaultdict(lambda: collections.defaultdict(list))
    
    all_oof_preds = []
    all_oof_targets = []
    indices_per_fold = []
    
    current_idx = 0

    # --- CROSS VALIDATION LOOP ---
    for fold in range(cfg.k_folds):
        print(f"\n--- FOLD {fold + 1}/{cfg.k_folds} ---")
        
        train_loader, val_loader = get_loaders_for_fold(fold_idx=fold+1, 
                                                        batch_size=cfg.batch_size, 
                                                        transform_test=transformation, 
                                                        transform_train=transformation,
                                                        num_workers=cfg.num_workers)

        model = WraperModel(num_classes=cfg.output_dim, feature_extraction=True)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        
        #class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_loader.dataset.targets), y=train_loader.dataset.targets)
        #weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        #criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        criterion = nn.CrossEntropyLoss()

        # For early stopping
        stopper = EarlyStopping(patience = cfg.patience, min_delta = cfg.min_delta)

        # Training Loop
        for epoch in tqdm.tqdm(range(cfg.num_epochs), desc=f"Fold {fold+1} Epochs"):
            _, _, train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            _, _, _, val_loss, val_acc = test(model, val_loader, criterion, device)
        
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Early stopping
            stopper(val_loss, model)
            if stopper.early_stop == True:
                print(f"Stopping training in epoch {epoch+1}")
                break

            # Store for aggregation
            history[epoch]['train_acc'].append(train_acc)
            history[epoch]['train_loss'].append(train_loss)
            history[epoch]['val_acc'].append(val_acc)
            history[epoch]['val_loss'].append(val_loss)

        # End of Fold
        # Save model weights
        if cfg.save_weights:
            torch.save(stopper.best_model, f"./saved_models/fold-{fold + 1}_{cfg.learning_rate}_{cfg.num_epochs}.pt")
        
        model.load_state_dict(stopper.best_model)
        val_pred, val_true, _, _, _ = test(model, val_loader, criterion, device)
        
        # Calculate val_ids for this fold
        num_samples = len(val_pred)
        val_ids = np.arange(current_idx, current_idx + num_samples)
        
        # Store data
        indices_per_fold.append(val_ids)
        all_oof_preds.extend(val_pred)
        all_oof_targets.extend(val_true)
        
        # Update the global index pointer
        current_idx += num_samples

        del train_loader, val_loader, model, optimizer, criterion
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    # --- END CROSS VALIDATION LOOP ---
        
    #################################################
    ############## METRICS CALCULATION ##############
    #################################################

    metrics = FoldMetrics(all_oof_targets, all_oof_preds, indices_per_fold)
    metrics_dict = metrics.compute()
    summary = metrics.get_summary()    
    wandb.log(metrics_dict)
    wandb.log(summary)
    
    fig_acc, fig_loss = metrics.get_training_plots(history)

    # Log Images to WandB
    wandb.log({
        "aggregated_accuracy_plot": wandb.Image(fig_acc),
        "aggregated_loss_plot": wandb.Image(fig_loss)
    })
    
    plt.close(fig_acc)
    plt.close(fig_loss)
    
    wandb.finish()
    
if __name__ == "__main__":
    run_experiment(experiment_config=None)
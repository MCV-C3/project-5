from typing import *
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2  as F
import tqdm
import wandb
from sklearn.model_selection import KFold, GroupKFold
import collections

from models import SimpleModel, SimpleCNN
from main import train, test, train_patch_based, test_patch_based
from metrics import FoldMetrics
from svm_utils import train_svm, train_svm_patches
from utils import PatchGrid
from fisher_vectors import FisherVectors

import os

def run_experiment(wandb_config=None, experiment_config=None):
    # Default configurations
    default_wandb_config = {
        "project": "C3-Week2-MLP",
        "entity": "project-5",
    }
    
    default_experiment_config = {
        "image_size": (224, 224),
        "batch_size": 256,
        "num_epochs": 20,
        "learning_rate": 0.001,
        "hidden_dim": 300,
        "encoding": "fisher",
        "pca":False,
        "output_dim": 11,
        "num_workers": 8,
        "num_hidden_layers":2,
        "k_folds": 5,
        "task_type": "mlp_svm", # mlp_svm, mlp_only
        "model_type": "mlp",
        "patches_lvl": 1,
        "patches_method":"vote",
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
    image_size = (224//2**cfg.patches_lvl, 224//2**cfg.patches_lvl) if cfg.patches_lvl > 0 else cfg.image_size
    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=image_size),
                                ])
    
    print("Loading datasets...")
    if cfg.patches_lvl > 0:
        data_train = ImageFolder(f"~/data/places_reduced_patches_lvl_{cfg.patches_lvl}/train", transform=transformation)
        
        # get groups based on patches
    else:
        data_train = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/train", transform=transformation)
    
    # group each image per patches if patches_lvl > 0
    if cfg.patches_lvl > 0:
        groups = []
        num_patches = 4 ** cfg.patches_lvl
        for idx in range(0, len(data_train), num_patches):
            groups.extend([idx for i in range(num_patches)])
        gkfold = GroupKFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)
    

    full_targets = np.array(data_train.targets)
    oof_preds_full = np.zeros(len(data_train), dtype=int)
    svm_oof_preds_full = np.zeros(len(data_train), dtype=int) if cfg.task_type == "mlp_svm" else None

    C, H, W = np.asarray(data_train[0][0]).shape
    
        
    print(f"Channels: {C}, Height: {H}, Width: {W}")
    input_dim = C * H * W
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indices_per_fold = []
    
    # --- HISTORY STORAGE ---
    # history[epoch]['val_acc'] = [fold1, fold2...]
    history = collections.defaultdict(lambda: collections.defaultdict(list))

    # --- CROSS VALIDATION LOOP ---
    kfold_split = gkfold.split(data_train, groups=groups) if cfg.patches_lvl > 0 else kfold.split(data_train)
    for fold, (train_ids, val_ids) in enumerate(kfold_split):
        indices_per_fold.append(val_ids)
        
        print(f"\n--- FOLD {fold + 1}/{cfg.k_folds} ---")
        
        train_sub = Subset(data_train, train_ids)
        val_sub = Subset(data_train, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True, 
                                  num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.num_workers, pin_memory=True)

        if cfg.model_type == 'cnn':
            # Get optional CNN-specific hyperparameters from config
            conv_stride = getattr(cfg, 'conv_stride', 2)
            kernel_size = getattr(cfg, 'kernel_size', 5)
            
            model = SimpleCNN(in_channels=C, hidden_channels=cfg.hidden_dim, output_d=cfg.output_dim, 
                              img_size=H, conv_stride=conv_stride, kernel_size=kernel_size)
        else:
            model = SimpleModel(input_d=input_dim, hidden_d=cfg.hidden_dim, output_d=cfg.output_dim, num_hidden_layers=cfg.num_hidden_layers)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        for epoch in tqdm.tqdm(range(cfg.num_epochs), desc=f"Fold {fold+1} Epochs"):
            _, _, _, train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            _, _, _, val_loss, val_acc = test(model, val_loader, criterion, device)
        
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            wandb.log({
                f"fold_{fold}_train_loss": train_loss,
                f"fold_{fold}_train_acc": train_acc,
                f"fold_{fold}_val_loss": val_loss,
                f"fold_{fold}_val_acc": val_acc,
                "epoch": epoch
            })

            # Store for aggregation
            history[epoch]['train_acc'].append(train_acc)
            history[epoch]['train_loss'].append(train_loss)
            history[epoch]['val_acc'].append(val_acc)
            history[epoch]['val_loss'].append(val_loss)

        # End of Fold
        final_val_preds, _, _, _, _ = test(model, val_loader, criterion, device)
        oof_preds_full[val_ids] = final_val_preds
        
        
        use_fisher = True if cfg.encoding == "fisher" else False
        use_pca = cfg.pca
            
            
        if cfg.task_type == "mlp_svm":
            if cfg.patches_lvl == 0:
                svm_preds, _, svm_acc = train_svm(train_loader, val_loader, model, device, fisher=use_fisher, pca=use_pca)
                svm_oof_preds_full[val_ids] = svm_preds
                wandb.log({"svm_fold_acc": svm_acc})
            
            else:
                svm_preds, _, svm_acc = train_svm_patches(train_loader, val_loader, model, device, patches_lvl=cfg.patches_lvl, method=cfg.patches_method, fisher=use_fisher, pca=use_pca)
                svm_preds_expanded = np.repeat(svm_preds, num_patches)
                svm_oof_preds_full[val_ids] = svm_preds_expanded
                wandb.log({"svm_fold_acc": svm_acc})
        
        del train_loader, val_loader, model, optimizer, criterion
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    # --- END CROSS VALIDATION LOOP ---
        
    #################################################
    ############## METRICS CALCULATION ##############
    #################################################
    
    # Choose which predictions to use for final metrics
    if cfg.task_type == "mlp_svm":
        # Use SVM predictions for metrics
        final_preds = svm_oof_preds_full
        metrics_prefix = "mlp_svm_"
    else:
        # Use MLP predictions for metrics
        final_preds = oof_preds_full
        metrics_prefix = "mlp_"
        
    if cfg.patches_lvl > 0:
        print("Converting results from Patch-level to Image-level...")        
        full_targets = full_targets[::num_patches]
        final_preds = final_preds[::num_patches]
        
        new_indices_per_fold = []
        for val_ids in indices_per_fold:
            img_ids = val_ids // num_patches
            new_indices_per_fold.append(np.unique(img_ids))
            
        indices_per_fold = new_indices_per_fold
        
        print(f"New shapes -> Targets: {full_targets.shape}, Preds: {final_preds.shape}")
    
    metrics = FoldMetrics(full_targets, final_preds, indices_per_fold)
    metrics_dict = metrics.compute()
    summary = metrics.get_summary()
    
    # Add prefix to metric names if using SVM
    if metrics_prefix:
        metrics_dict = {f"{metrics_prefix}{k}": v for k, v in metrics_dict.items()}
        summary = {f"{metrics_prefix}{k}": v for k, v in summary.items()}
    
    wandb.log(metrics_dict)
    wandb.log(summary)
    
    fig_acc, fig_loss = metrics.get_training_plots(history)

    # Log Images to WandB
    wandb.log({
        f"{metrics_prefix}aggregated_accuracy_plot": wandb.Image(fig_acc),
        f"{metrics_prefix}aggregated_loss_plot": wandb.Image(fig_loss)
    })
    
    plt.close(fig_acc)
    plt.close(fig_loss)    
    
if __name__ == "__main__":
    patches_lvl = 2
    image_size = (224//2**patches_lvl, 224//2**patches_lvl)
    experiment_config = {
        "image_size": image_size,
        "batch_size": 1024,
        "num_epochs": 20,
        "learning_rate": 0.001,
        "hidden_dim": 300,
        "output_dim": 11,
        "encoding": "fisher",
        "pca":True,
        "num_workers": 16, # 8
        "k_folds": 3,
        "task_type": "mlp_svm", # mlp_svm, mlp_only
        "model_type": "mlp",
        "patches_lvl": patches_lvl,
        "patches_method":"sum"
    }
    run_experiment(experiment_config=experiment_config)
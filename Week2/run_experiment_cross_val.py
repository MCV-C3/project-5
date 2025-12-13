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
from sklearn.model_selection import KFold
import collections

from models import SimpleModel
from main import train, test
from metrics import FoldMetrics
from svm_utils import train_svm

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
        "output_dim": 11,
        "num_workers": 8,
        "k_folds": 5,
        "task_type": "mlp_only",
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
    
    print("Loading datasets...")
    # NOTE: Check your path
    data_train = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/train", transform=transformation)
    
    kfold = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)

    full_targets = np.array(data_train.targets)
    oof_preds_full = np.zeros(len(data_train), dtype=int)
    svm_oof_preds_full = np.zeros(len(data_train), dtype=int) if cfg.task_type == "mlp_svm" else None
    
    C, H, W = np.asarray(data_train[0][0]).shape
    input_dim = C * H * W
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indices_per_fold = []
    
    # --- HISTORY STORAGE ---
    # history[epoch]['val_acc'] = [fold1, fold2...]
    history = collections.defaultdict(lambda: collections.defaultdict(list))

    # --- CROSS VALIDATION LOOP ---
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data_train)):
        indices_per_fold.append(val_ids)
        
        print(f"\n--- FOLD {fold + 1}/{cfg.k_folds} ---")
        
        train_sub = Subset(data_train, train_ids)
        val_sub = Subset(data_train, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True, 
                                  num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.num_workers, pin_memory=True)

        model = SimpleModel(input_d=input_dim, hidden_d=cfg.hidden_dim, output_d=cfg.output_dim)
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
        
        if cfg.task_type == "mlp_svm":
            svm_preds, _, svm_acc = train_svm(train_loader, val_loader, model, device)
            svm_oof_preds_full[val_ids] = svm_preds
            wandb.log({"svm_fold_acc": svm_acc})
        
        del train_loader, val_loader, model, optimizer, criterion
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    # --- END CROSS VALIDATION LOOP ---
        
    #################################################
    ############## METRICS CALCULATION ##############
    #################################################
    
    metrics_mlp = FoldMetrics(full_targets, oof_preds_full, indices_per_fold)
    metrics_mlp_dict = metrics_mlp.compute()
    summary = metrics_mlp.get_summary()
    wandb.log(metrics_mlp_dict)
    wandb.log(summary)
    
    fig_acc, fig_loss = metrics_mlp.get_training_plots(history)

    # 4. Log Images to WandB
    wandb.log({
        "aggregated_accuracy_plot": wandb.Image(fig_acc),
        "aggregated_loss_plot": wandb.Image(fig_loss)
    })
    
    plt.close(fig_acc)
    plt.close(fig_loss)
    
    if cfg.task_type == "mlp_svm":
        metrics_svm = FoldMetrics(full_targets, svm_oof_preds_full, indices_per_fold)
        metrics_svm_dict = metrics_svm.compute()
        wandb.log({f"svm_{k}": v for k, v in metrics_svm_dict.items()})    
    
if __name__ == "__main__":
    run_experiment()
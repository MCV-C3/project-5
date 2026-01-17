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

def get_optimizer (model, cfg):
    params = model.parameters()
    lr = cfg.learning_rate
    wd = cfg.weight_decay # Regularizer (L2 penalty)
    mom = cfg.momentum

    opt_name = cfg.optimizer.lower()

    if opt_name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, momentum=mom, weight_decay=wd)
    elif opt_name == 'adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=wd)
    elif opt_name == 'adadelta':
        return optim.Adadelta(params, lr=lr, weight_decay=wd)
    elif opt_name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == 'adamax':
        return optim.Adamax(params, lr=lr, weight_decay=wd)
    elif opt_name == 'nadam':
        return optim.NAdam(params, lr=lr, weight_decay=wd)
    else:
        print(f"Optimizer {opt_name} not found. Defaulting to Adam.")
        return optim.Adam(params, lr=lr, weight_decay=wd)

def get_loaders_for_fold(fold_idx, batch_size=32, transform_train=None, transform_test=None, num_workers=8):
    fold_dir = f"MIT_small_train_{fold_idx}"
    
    train_path = os.path.join(BASE_PATH, fold_dir, "train")
    test_path = os.path.join(BASE_PATH, fold_dir, "test")
    
    train_dataset = ImageFolder(train_path, transform=transform_train)
    test_dataset = ImageFolder(test_path, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_augmented_loaders(fold_idx, cfg, aug_options, transform_base, norm_transform):
    """
    Assembles DataLoaders for the current fold using pre-defined augmentation options.
    """
    fold_dir = f"MIT_small_train_{fold_idx}"
    train_path = os.path.join(BASE_PATH, fold_dir, "train")
    test_path = os.path.join(BASE_PATH, fold_dir, "test")

    # --- LOGIC 1: Standard Augmentation (add_aug == False) ---
    if not cfg.get("add_aug", False):
        # Flatten all transforms from the dictionary into one sequential pipeline
        # These will use the specific probabilities (e.g., p=0.5) from your config
        active_augs = [op for ops_list in aug_options.values() for op in ops_list]
        
        # Build the single training transform
        # We add Resize at the end if Zoom (RandomResizedCrop) wasn't selected
        has_zoom = "zoom" in aug_options
        resize_step = [] if has_zoom else [F.Resize(cfg.image_size)]
        
        train_transform = F.Compose(transform_base + active_augs + resize_step + norm_transform)
        train_datasets = [ImageFolder(train_path, transform=train_transform)]
        print(f"   > [Fold {fold_idx}] Standard Mode: Applying random augs to original dataset.")

    # --- LOGIC 2: Dataset Expansion (add_aug == True) ---
    else:
        # Original dataset stays "clean" (No random augs applied)
        clean_trans = F.Compose(transform_base + [F.Resize(cfg.image_size)] + norm_transform)
        train_datasets = [ImageFolder(train_path, transform=clean_trans)]
        print(f"   > [Fold {fold_idx}] Expanded Mode: Base dataset is clean.")

        # Create a separate copy for EACH transformation with 100% probability
        for name, ops in aug_options.items():
            # Zoom handles its own resizing
            resize_step = [] if name == "zoom" else [F.Resize(cfg.image_size)]
            
            # Assembly: Base -> Transform(s) at p=1.0 -> Resize -> Norm
            full_aug_trans = F.Compose(transform_base + ops + resize_step + norm_transform)
            train_datasets.append(ImageFolder(train_path, transform=full_aug_trans))
            print(f"   > [Fold {fold_idx}] Added 100% augmented copy: {name}")
    # --- FINAL ASSEMBLY ---
    # ConcatDataset handles both cases: a list of 1 (standard) or a list of many (expanded)
    combined_train_ds = torch.utils.data.ConcatDataset(train_datasets)

    # Validation always uses clean transform
    val_trans = F.Compose(transform_base + [F.Resize(cfg.image_size)] + norm_transform)
    val_ds = ImageFolder(test_path, transform=val_trans)

    train_loader = DataLoader(
        combined_train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers
    )
    
    total_imagenes = len(train_loader.dataset)
    print(f"Número total de imágenes: {total_imagenes}")

    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False
    )

    print(f"   > Total Training Samples: {len(combined_train_ds)}")

    return train_loader, val_loader

def run_experiment(wandb_config=None, experiment_config=None):
    # Default configurations
    default_wandb_config = {
        "project": "C3-Week3-MobileNet",
        "entity": "marc-org",
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
        "k_folds": 4,
        "aug_horizontal_flip": 0.0,
        "aug_rotation": 0.0,
        "aug_color_jitter": 0.0,
        "aug_zoom": 0.0,
        "aug_gaussian_blur": 0.0,
        "use_imagenet_norm": False,
        "add_aug": False,
        "optimizer": "Adam",
        "momentum": 0.9,
        "weight_decay": 0.0,
        "dropout_prob": 0.2, # Topology default
        "blocks_to_keep": list(range(14)),
        "out_feat": -1,
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

    base_transforms = [
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
    ]
    norm_transform = [F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] if cfg.use_imagenet_norm else []

    p_val = lambda specific_p: max(int(cfg.add_aug), specific_p)

    aug_options = {}
    if cfg.aug_horizontal_flip > 0:
        aug_options['flip'] = [F.RandomHorizontalFlip(p=p_val(cfg.aug_horizontal_flip))]
    if cfg.aug_rotation > 0:
        aug_options['rot'] = [F.RandomApply([F.RandomRotation(45)], p=p_val(cfg.aug_rotation))]
    if cfg.aug_color_jitter > 0:
        aug_options['jit'] = [F.RandomApply([F.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=p_val(cfg.aug_color_jitter))]
    if cfg.aug_zoom > 0:
        aug_options['zoom'] = [F.RandomApply([F.RandomResizedCrop(cfg.image_size, scale=(0.75, 1.0))], p=p_val(cfg.aug_zoom))]
    if cfg.aug_gaussian_blur > 0:
        aug_options['blur'] = [F.RandomApply([F.GaussianBlur((5, 9))], p=p_val(cfg.aug_gaussian_blur))]    # 3. Normalization (Applied last)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- HISTORY STORAGE ---
    history = collections.defaultdict(lambda: collections.defaultdict(list))
    
    all_oof_preds = []
    all_oof_targets = []
    indices_per_fold = []
    
    current_idx = 0

    # --- CROSS VALIDATION LOOP ---
    for fold in range(cfg.k_folds):
        print(f"\n--- FOLD {fold + 1}/{cfg.k_folds} ---")
        train_loader, val_loader = get_augmented_loaders(
            fold_idx=fold+1, 
            cfg=cfg, 
            aug_options=aug_options,
            transform_base=base_transforms, 
            norm_transform=norm_transform
        )
  
        if cfg.get("backbone") == "vit" or cfg.get("backbone") == "swin": 
            print("Using ViT/Swin Backbone")
            model = WraperModel(
                            num_classes=cfg.output_dim,
                            feature_extraction=cfg.feature_extraction,
                            dropout_prob=cfg.dropout_prob,
                            backbone_name=cfg.backbone,
                        )
        elif cfg.get("backbone") == "mobilenet_v2":
            print("Using MobileNetV2 Backbone")
            model = WraperModel(
                            num_classes=cfg.output_dim,
                            feature_extraction=cfg.feature_extraction,
                            blocks_to_keep=cfg.blocks_to_keep,
                            out_feat=cfg.out_feat,
                            dropout_prob=cfg.dropout_prob,
                            backbone_name=cfg.backbone,
                        )
        else:
            raise ValueError(f"Backbone {cfg.get('backbone')} not supported.")

        model = model.to(device)
        
        optimizer = get_optimizer(model=model, cfg=cfg)
        
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
    default_experiment_config = {
        "image_size": (224, 224),
        "batch_size": 256,
        "num_epochs": 20,
        "learning_rate": 0.001,
        "output_dim": 11,
        "num_workers": 8,
        "patience": 5,
        "min_delta": 0,
        "save_weights": True,
        "k_folds": 4
    }
    experiments = [
        {"experiment_name": "baseline"}, # All augs at 0.0
        {"experiment_name": "aug_flip", "aug_horizontal_flip": 0.5},
        {"experiment_name": "aug_rotation", "aug_rotation": 0.3},
        {"experiment_name": "aug_jitter", "aug_color_jitter": 0.2},
        {"experiment_name": "aug_zoom", "aug_zoom": 0.2},
        {"experiment_name": "aug_blur", "aug_gaussian_blur": 0.2},
        {"experiment_name": "imagenet_norm", "use_imagenet_norm":True}
    ]


    for exp_cfg in experiments:
        config = {**default_experiment_config, **exp_cfg}
        print(f"\n🚀 LAUNCHING EXPERIMENT: {config['experiment_name']}")
        run_experiment(experiment_config=config)

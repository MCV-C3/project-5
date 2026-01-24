from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as t_F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as F
import tqdm
import wandb
from sklearn.utils.class_weight import compute_class_weight
import collections
import os

from models import MCV_Net, EarlyStopping
from main import train, test
from metrics import FoldMetrics, compute_distance, compute_efficiency_ratio_metric, get_model_parameters

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_PATH = "../../../mcv/datasets/C3/2425/"

# wandb.login(key="X"*40)
# os.environ["WANDB_MODE"] = "offline"
# wandb.disabled = True
# --- 1. DISTILLATION TRAINING FUNCTION ---
def train_distill(student, teacher, train_loader, optimizer, device, T=4.0, alpha=0.5):
    """
    Train student using both Hard Labels (Task Loss) and Soft Labels (Teacher).
    """
    student.train()
    teacher.eval() # Teacher must be frozen
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    criterion_task = nn.CrossEntropyLoss()
    criterion_distill = nn.KLDivLoss(reduction="batchmean")
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Teacher Forward (No Grad)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # Student Forward
        student_logits = student(inputs)
        
        # 1. Hard Loss (Ground Truth)
        loss_hard = criterion_task(student_logits, labels)
        
        # 2. Soft Loss (Distillation)
        student_soft = t_F.log_softmax(student_logits / T, dim=1)
        teacher_soft = t_F.softmax(teacher_logits / T, dim=1)
        
        loss_distill = criterion_distill(student_soft, teacher_soft)
        
        # Combined Loss
        loss_soft = loss_distill * (T**2)

	
        loss = alpha * loss_hard + (1 - alpha) * loss_soft
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(student_logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# --- 2. MODEL LOADERS ---
def get_dinov3_convnext(model_size='tiny', num_classes=8, device='cuda'):
    """Loads DINOv3 ConvNeXt from official repo."""
    model_names = {
        'tiny': 'dinov3_convnext_tiny', 'small': 'dinov3_convnext_small',
        'base': 'dinov3_convnext_base', 'large': 'dinov3_convnext_large'
    }
    print(f"Loading {model_names[model_size]} from facebookresearch/dinov3...")
    
    repo = 'facebookresearch/dinov3'
    model = torch.hub.load(repo, model_names[model_size], pretrained=True)
    
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(768, num_classes)

    return model.to(device)

def get_model_for_experiment(model_type, cfg, num_classes, device):
    """Generic model selector."""
    if 'convnext' in model_type:
        convnext_size = model_type.split('_')[-1]
        
        if 'dinov3' in model_type:
            model = get_dinov3_convnext(convnext_size, num_classes, device)
        else:
            # Standard ImageNet ConvNeXt
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            print(f"Loading Standard ConvNeXt-{convnext_size}...")
            # Assuming Tiny for brevity, expand if needed
            model = convnext_tiny(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
            model = model.to(device)
            
    else: 
        # Default: Custom MCV_Net
        print("Loading Custom MCV_Net...")
        model = MCV_Net(image_size=cfg.image_size, block_type=cfg.block_type, 
                        init_chan=cfg.init_chan, num_blocks=cfg.num_blocks, 
                        filters=cfg.filters, units_fc=cfg.units_fc,
                        dropout_prob_1=cfg.dropout_prob_1, 
                        dropout_prob_2=cfg.dropout_prob_2,
                        num_classes=num_classes)
        model = model.to(device)

    return model

# --- 3. UTILS & LOADERS ---
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
    """K-Fold Loader"""
    fold_dir = f"MIT_small_train_{fold_idx}"
    
    train_path = os.path.join(BASE_PATH, fold_dir, "train")
    test_path = os.path.join(BASE_PATH, fold_dir, "test")
    
    train_dataset = ImageFolder(train_path, transform=transform_train)
    test_dataset = ImageFolder(test_path, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# --- 4. MAIN EXPERIMENT LOOP ---
def run_experiment(wandb_config=None, experiment_config=None):
    # Defaults
    default_wandb_config = {
        "project": "C3-Week3-MobileNet",
        "entity": "marc-org",
    }
    default_experiment_config = {
        "image_size": (224, 224), "batch_size": 32, "num_epochs": 20, "learning_rate": 0.001,
        "num_workers": 8, "patience": 5, "min_delta": 0, "save_weights": True, "k_folds": 4,
        "model_type": "mcv", 
        "optimizer": "Adam", "filters": [15,30,45,60,75], "units_fc": 64,
        "distill": True, # ENABLE DISTILLATION
        "teacher_model_type": "convnext_tiny",
        "teacher_weights": "./saved_models/convnext_tiny_MIT_large_train.pt", # CHECK THIS PATH
        "temperature": 4.0,
        "alpha": 0.5
    }
    
    # Merge Configs
    if wandb_config is None: wandb_config = default_wandb_config
    else: wandb_config = {**default_wandb_config, **wandb_config}
    
    if experiment_config is None: experiment_config = default_experiment_config
    else: experiment_config = {**default_experiment_config, **experiment_config}

    # experiment_config = {**default_experiment_config, **(experiment_config or {})}
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        config=experiment_config,
        mode="offline"
    )
    
    cfg = wandb.config
    print(f"Experiment: {cfg.model_type} | Distill: {cfg.distill} | K-Folds: {cfg.k_folds}")
    print(f"Config: {dict(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Transforms
    test_transform = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=cfg.image_size),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if cfg.data_aug:
        train_transform = F.Compose([
            F.ToImage(),
            
            F.RandomResizedCrop(size=cfg.image_size, scale=(0.85, 1.0), antialias=True),
            F.RandomHorizontalFlip(p=0.5),
            # F.RandomRotation(degrees=20),
            F.RandomApply([F.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
            # F.RandomApply([F.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.15),
            
            F.ToDtype(torch.float32, scale=True),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = test_transform

    # --- PREPARE TEACHER (Loaded Once, reused across folds to save memory/time) ---
    # In a perfect world, you'd load a different teacher per fold. 
    # Here we use the "Universal Teacher" you trained previously.
    teacher_model = None
    if cfg.distill:
        print(f"Loading Teacher: {cfg.teacher_model_type}")
        # Need dummy classes init, will fix later
        teacher_model = get_model_for_experiment(cfg.teacher_model_type, cfg, 8, device) 
        
        if cfg.teacher_weights and os.path.exists(cfg.teacher_weights):
            print(f"Loading Teacher weights from {cfg.teacher_weights}")
            teacher_model.load_state_dict(torch.load(cfg.teacher_weights))
        else:
            raise ValueError(f"Teacher weights not found at: {cfg.teacher_weights}")
        
        teacher_model.eval()
        for p in teacher_model.parameters(): p.requires_grad = False

    # Metrics Storage
    history = collections.defaultdict(lambda: collections.defaultdict(list))
    all_oof_preds, all_oof_targets, indices_per_fold = [], [], []
    current_idx = 0
    num_params = None

    # --- K-FOLD LOOP ---
    for fold in range(cfg.k_folds):
        print(f"\n--- FOLD {fold + 1}/{cfg.k_folds} ---")
        
        # 1. Get Data
        train_loader, val_loader = get_loaders_for_fold(
            fold_idx=fold+1, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers,
            transform_train=train_transform,
            transform_test=test_transform
        )
        num_classes = len(train_loader.dataset.classes)
        
        # 2. Init New Student
        student_model = get_model_for_experiment(cfg.model_type, cfg, num_classes, device)
        optimizer = get_optimizer(student_model, cfg)
        criterion = nn.CrossEntropyLoss()
        
        if num_params is None:
            num_params = get_model_parameters(student_model)
            print(f"Student Params: {num_params:,}")

        # 3. Training Loop
        stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

        for epoch in tqdm.tqdm(range(cfg.num_epochs), desc=f"Fold {fold+1}"):
            # Distill or Standard Train
            if cfg.distill and teacher_model:
                train_loss, train_acc = train_distill(
                    student_model, teacher_model, train_loader, optimizer, device, 
                    T=cfg.temperature, alpha=cfg.alpha
                )
            else:
                _, _, _, train_loss, train_acc = train(student_model, train_loader, criterion, optimizer, device)

            # Validation
            _, _, _, val_loss, val_acc = test(student_model, val_loader, criterion, device)
            
            print(f"Ep {epoch+1} | T_Loss: {train_loss:.4f} T_Acc: {train_acc:.4f} | V_Loss: {val_loss:.4f} V_Acc: {val_acc:.4f}")
            
            # Store history (averaged across folds later)
            history[epoch]['train_acc'].append(train_acc)
            history[epoch]['train_loss'].append(train_loss)
            history[epoch]['val_acc'].append(val_acc)
            history[epoch]['val_loss'].append(val_loss)


            stopper(val_loss, student_model)
            if stopper.early_stop:
                print(f"Early Stopping Fold {fold+1}")
                break
        if cfg.save_weights:
            torch.save(stopper.best_model, f"./saved_models/fold_{fold+1}_{cfg.block_type}.pt")
        # 4. Finish Fold
        student_model.load_state_dict(stopper.best_model)
        val_pred, val_true, _, _, _ = test(student_model, val_loader, criterion, device)
        
        # Store for Metrics
        num_samples = len(val_pred)
        indices_per_fold.append(np.arange(current_idx, current_idx + num_samples))
        all_oof_preds.extend(val_pred)
        all_oof_targets.extend(val_true)
        current_idx += num_samples

        # Cleanup Student (Keep Teacher)
        del student_model, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()

    # --- METRICS & WANDB ---
    metrics = FoldMetrics(all_oof_targets, all_oof_preds, indices_per_fold)
    metrics_dict = metrics.compute()
    summary = metrics.get_summary()

    summary.update({
        'num_parameters': num_params,
        'params_in_100k': num_params / 10**5,
        'efficiency_ratio_metric': compute_efficiency_ratio_metric(summary['fold_accuracy_mean'], num_params),
        'distance': compute_distance(summary['fold_accuracy_mean'], num_params)
    })

    wandb.log(metrics_dict)
    wandb.log(summary)
    
    fig_acc, fig_loss = metrics.get_training_plots(history)
    wandb.log({"acc_plot": wandb.Image(fig_acc), "loss_plot": wandb.Image(fig_loss)})
    
    plt.close(fig_acc)
    plt.close(fig_loss)
    wandb.finish()

if __name__ == "__main__":
    # CONFIGURATION FOR DISTILLATION EXPERIMENT
    config = {
        "model_type": "mcv",          # The Student
        "distill": True,              # Enable Distillation
        "teacher_model_type": "convnext_tiny",
        "teacher_weights": "./saved_models/fold_1_convnext_tiny_MIT_Large.pt",
        "learning_rate": 0.001,       # Standard LR for student
        "k_folds": 4,                 # 4-Fold CV
        "num_epochs": 20
    }
    
    print("🚀 Launching 4-Fold Distillation Experiment...")
    run_experiment(experiment_config=config)

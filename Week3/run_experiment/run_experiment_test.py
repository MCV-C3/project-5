from typing import *
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import wandb

from models import WraperModel, EarlyStopping
from main import train, test, plot_computational_graph, plot_metrics
from metrics import TestMetrics

import os

from torch.utils.data import ConcatDataset
import os

def get_optimizer(model, cfg):
    params = model.parameters()
    lr = cfg.learning_rate
    wd = cfg.get("weight_decay", 0.0)
    mom = cfg.get("momentum", 0.9)
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

def get_augmented_datasets(train_path: str, test_path: str, cfg, image_size: int):
    train_path = os.path.expanduser(train_path)
    test_path = os.path.expanduser(test_path)
    
    norm_transform = []
    if getattr(cfg, "use_imagenet_norm", False):
        norm_transform = [F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    # ---- Base transforms (always applied) ----
    base_transform = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize((image_size, image_size)),
    ] + norm_transform)

    test_transform = base_transform

    # ---- Probability logic ----
    p_val = lambda specific_p: max(int(cfg.add_aug), specific_p)

    aug_options = {
        "horizontal_flip": (
            [F.RandomHorizontalFlip(p=p_val(cfg.aug_horizontal_flip))]
            if cfg.aug_horizontal_flip > 0 else []
        ),

        "rotation": (
            [F.RandomApply(
                [F.RandomRotation(45)],
                p=p_val(cfg.aug_rotation)
            )]
            if cfg.aug_rotation > 0 else []
        ),

        "color_jitter": (
            [F.RandomApply(
                [F.ColorJitter(0.2, 0.2, 0.2, 0.1)],
                p=p_val(cfg.aug_color_jitter)
            )]
            if cfg.aug_color_jitter > 0 else []
        ),

        "zoom": (
            [F.RandomApply(
                [F.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1)
                )],
                p=p_val(cfg.aug_zoom)
            )]
            if cfg.aug_zoom > 0 else []
        ),

        "gaussian_blur": (
            [F.RandomApply(
                [F.GaussianBlur((5, 9))],
                p=p_val(cfg.aug_gaussian_blur)
            )]
            if cfg.aug_gaussian_blur > 0 else []
        ),
    }


    # ---- Active augmentations ----
    active_augs = []

    if cfg.aug_horizontal_flip:
        active_augs.append(aug_options["horizontal_flip"])

    if cfg.aug_rotation:
        active_augs.append(aug_options["rotation"])

    if cfg.aug_color_jitter:
        active_augs.append(aug_options["color_jitter"])

    if cfg.aug_zoom:
        active_augs.append(aug_options["zoom"])

    if cfg.aug_gaussian_blur:
        active_augs.append(aug_options["gaussian_blur"])
        
    # Flatten augmentation dict into a list of transforms
    active_augs = []
    for aug_list in aug_options.values():
        active_augs.extend(aug_list)


    # ---- Training dataset ----
    if cfg.add_aug:
        # Dataset expansion mode (p forced to 1)
        datasets = []

        # Original dataset
        datasets.append(
            ImageFolder(train_path, transform=base_transform)
        )

        # One dataset per augmentation
        for aug in active_augs:
            aug_transform = F.Compose([
                base_transform,
                aug
            ])
            datasets.append(
                ImageFolder(train_path, transform=aug_transform)
            )

        train_dataset = ConcatDataset(datasets)

    else:
        # Random augmentation mode
        if len(active_augs) > 0:
            train_transform = F.Compose([
                base_transform,
                *active_augs
            ])
        else:
            train_transform = base_transform

        train_dataset = ImageFolder(
            train_path,
            transform=train_transform
        )

    # ---- Test dataset (NO augmentation) ----
    test_dataset = ImageFolder(
        test_path,
        transform=test_transform
    )

    DEBUG_AUGS = False

    if DEBUG_AUGS:
        os.makedirs("debug_augs", exist_ok=True)

        # Get a raw sample (NO augmentation)
        raw_dataset = ImageFolder(
            train_path,
            transform=base_transform
        )

        img, label = raw_dataset[32]   # img: Tensor [C, H, W]

        # Save original
        save_image(img, "debug_augs/original.png")

        print("Saved: original.png")

        # Save each augmentation independently
        for aug_name, aug_list in aug_options.items():
            for i, aug in enumerate(aug_list):
                augmented_img = aug(img)

                out_path = f"debug_augs/{aug_name}_{i}.png"
                save_image(augmented_img, out_path)

                print(f"Saved: {out_path}")

        print("\n✅ Augmentation debug images saved. Stopping execution.")
        exit()
    
    return train_dataset, test_dataset


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
        "add_aug": False
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
        config=experiment_config
    )

    cfg = wandb.config
    
    # Normalize image_size to int
    if isinstance(cfg.image_size, (list, tuple)):
        image_size = cfg.image_size[0]
    else:
        image_size = cfg.image_size


    print(f"Starting experiment with config: {cfg}")
    
    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=image_size),
                                    #F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    # Get dataset name from config or use default
    dataset_name = getattr(cfg, 'dataset_name', 'MIT_large_train')
    
    data_train, data_test = get_augmented_datasets(
        train_path=f"~/mcv/datasets/C3/2425/{dataset_name}/train",
        test_path=f"~/mcv/datasets/C3/2425/{dataset_name}/test",
        cfg=cfg,
        image_size=image_size
    )


    train_loader = DataLoader(data_train, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(data_test, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, num_workers=cfg.num_workers)

    # C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model to be used
    if cfg.get("backbone") == "vit" or cfg.get("backbone") == "swin": 
        print(f"Using {cfg.get('backbone').upper()} Backbone")
        model = WraperModel(
                        num_classes=cfg.output_dim,
                        feature_extraction=cfg.get("feature_extraction", True),
                        dropout_prob=cfg.get("dropout_prob", 0.2),
                        backbone_name=cfg.backbone,
                    )
    elif cfg.get("backbone") == "mobilenet_v2":
        print("Using MobileNetV2 Backbone")
        model = WraperModel(
                        num_classes=cfg.output_dim,
                        feature_extraction=cfg.get("feature_extraction", True),
                        blocks_to_keep=cfg.get("blocks_to_keep", list(range(14))),
                        out_feat=cfg.get("out_feat", -1),
                        dropout_prob=cfg.get("dropout_prob", 0.2),
                        backbone_name=cfg.backbone,
                    )
    else:
        # Default fallback if backbone is not specified in config
        print("Backbone not specified or unknown, defaulting to MobileNetV2")
        model = WraperModel(num_classes=cfg.output_dim, feature_extraction=True)    
    
    model = model.to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, cfg)
    NUM_EPOCHS = cfg.num_epochs
    
    # For early stopping
    stopper = EarlyStopping(patience = cfg.patience, min_delta = cfg.min_delta)
    
    for epoch in tqdm.tqdm(range(NUM_EPOCHS), desc="TRAINING THE MODEL"):
        _, _, train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        _, _, _, test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_accuracy:.4f}")
        
        # Early stopping
        stopper(test_loss, model)
        if stopper.early_stop == True:
            print(f"Stopping training in epoch {epoch+1}")
            break

        # History
        wandb.log({
            "train_accuracies": train_accuracy,
            "test_accuracies": test_accuracy,
            "train_losses": train_loss,
            "test_losses": test_loss,
        })
        
    # Save model weights
    if cfg.save_weights:
        torch.save(stopper.best_model, f"./saved_models/test_{cfg.learning_rate}_{cfg.num_epochs}.pt")
    
    model.load_state_dict(stopper.best_model)
    test_pred, test_true, test_probs, _, _ = test(model, test_loader, criterion, device)

    # Calculate metrics
    metrics = TestMetrics(test_true, test_pred, test_probs)
    metrics_dict = metrics.compute()
    wandb.log(metrics_dict)
    
    #CONFMAT
    confmat_fig = metrics.plot_confusion_matrix()
    wandb.log({"confusion_matrix": wandb.Image(confmat_fig)})
    
    # ROC CURVE
    roc_fig = metrics.plot_roc_curve()
    wandb.log({"ROC_curve": wandb.Image(roc_fig)})
    
    wandb.finish()

if __name__ == "__main__":
    run_experiment(experiment_config=None)
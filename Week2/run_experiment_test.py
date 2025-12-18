from typing import *
from torch.utils.data import DataLoader, Subset
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

from models import SimpleModel, SimpleCNN
from main import train, test, plot_computational_graph, plot_metrics
from metrics import TestMetrics
from svm_utils import train_svm, train_svm_patches

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
        "k_folds": 5,
        "task_type": "mlp_svm", # mlp_svm, mlp_only
        "model_type": "mlp",
        "patches_lvl": 1,
        "patches_method":"vote",
    }
    
    # Merge with provided configs
    if wandb_config is None:
        wandb_config = default_wandb_config
    else:
        wandb_config = {**default_wandb_config, **wandb_config}
    
    if experiment_config is None:
        experiment_config = default_experiment_config
    else:
        experiment_config = {**default_experiment_config, **experiment_config}

    # Init wandb
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        config=experiment_config
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
        data_test = ImageFolder(f"~/data/places_reduced_patches_lvl_{cfg.patches_lvl}/test", transform=transformation)
        num_patches = 4 ** cfg.patches_lvl
    else:
        data_train = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/train", transform=transformation)
        data_test = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/val", transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(data_test, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, num_workers=cfg.num_workers)
    
    # Get Input Dimensions
    
    full_targets = np.array(data_train.targets)

    C, H, W = np.asarray(data_train[0][0]).shape
    input_dim = C * H * W
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model_type == 'cnn':
        # img_size must be the side length (assumes square resize)
        # Get optional CNN-specific hyperparameters from config
        conv_stride = getattr(cfg, 'conv_stride', 2)
        kernel_size = getattr(cfg, 'kernel_size', 5)
        
        model = SimpleCNN(in_channels=C, hidden_channels=cfg.hidden_dim, output_d=cfg.output_dim, 
                          img_size=H, conv_stride=conv_stride, kernel_size=kernel_size)
    else:
        model = SimpleModel(input_d=input_dim, hidden_d=cfg.hidden_dim, output_d=cfg.output_dim)
    #plot_computational_graph(model, input_size=(1, C*H*W))  # Batch size of 1, input_dim=10

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    NUM_EPOCHS = cfg.num_epochs

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for epoch in tqdm.tqdm(range(NUM_EPOCHS), desc="TRAINING THE MODEL"):
        _, _, _, train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        _, _, _, test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        #print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
        #      f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
        #      f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        wandb.log({
            f"train_loss": train_loss,
            f"train_acc": train_accuracy,
            f"test_loss": test_loss,
            f"test_acc": test_accuracy,
            "epoch": epoch
        })
        
    final_test_pred, final_test_true, final_test_probs, _, _ = test(model, test_loader, criterion, device)
    
    use_fisher = True if cfg.encoding == "fisher" else False
    use_pca = cfg.pca
    
    if cfg.task_type == "mlp_svm":
        if cfg.patches_lvl == 0:
            svm_preds, _, svm_acc = train_svm(train_loader, test_loader, model, device, fisher=use_fisher, pca=use_pca)
            wandb.log({"svm_fold_acc": svm_acc})
        else:
            print("train_svm_patches")
            svm_preds, _, svm_acc = train_svm_patches(train_loader, test_loader, model, device, patches_lvl=cfg.patches_lvl, method=cfg.patches_method, fisher=use_fisher, pca=use_pca)


    # Plot results
    #plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss")
    #plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy")
    
    # Choose which predictions to use for final metrics
    if cfg.task_type == "mlp_svm":
        # Use SVM predictions for metrics
        final_preds = svm_preds
        metrics_prefix = "mlp_svm_"
    else:
        # Use MLP predictions for metrics
        final_preds = final_test_pred
        metrics_prefix = "mlp_"
    
    if cfg.patches_lvl > 0:
        print("Converting results from Patch-level to Image-level...")
        final_test_probs = np.array(final_test_probs)
                
        full_targets = full_targets[::num_patches]
        if cfg.task_type != "mlp_svm":
            final_preds = final_preds[::num_patches]
        final_test_true = final_test_true[::num_patches]
        
        probs_grouped = final_test_probs.reshape(-1, num_patches, final_test_probs.shape[1])
        final_test_probs = np.sum(probs_grouped, axis=1)
        final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)
        
        # new_indices_per_fold = []
        # for val_ids in indices_per_fold:
        #     img_ids = val_ids // num_patches
        #     new_indices_per_fold.append(np.unique(img_ids))
            
        # indices_per_fold = new_indices_per_fold
        
        print(f"New shapes -> Targets: {full_targets.shape}, Preds: {final_preds.shape}")    
    
    # Calculate metrics
    metrics = TestMetrics(final_test_true, final_preds, final_test_probs)
    metrics_dict = metrics.compute()
    
    # Add prefix to metric names if using SVM
    if metrics_prefix:
        metrics_dict = {f"{metrics_prefix}{k}": v for k, v in metrics_dict.items()}

    wandb.log(metrics_dict)
    
    #CONFMAT
    confmat_fig = metrics.plot_confusion_matrix()
    wandb.log({"confusion_matrix": wandb.Image(confmat_fig)})
    
    # ROC CURVE
    roc_fig = metrics.plot_roc_curve()
    wandb.log({"ROC_curve": wandb.Image(roc_fig)})
    
    wandb.finish()
        
if __name__ == "__main__":
    patches_lvl = 1
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
        "k_folds": 2,
        "task_type": "mlp_svm", # mlp_svm, mlp_only
        "model_type": "mlp",
        "patches_lvl": patches_lvl,
        "patches_method":"sum"
    }
    run_experiment(experiment_config=experiment_config)
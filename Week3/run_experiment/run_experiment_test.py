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

from models import WraperModel, EarlyStopping
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
        "output_dim": 11,
        "num_workers": 8,
        "patience": 5,
        "min_delta": 0,
        "save_weights": False
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

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=cfg.image_size),
                                ])
    
    data_train = ImageFolder("~/data/Master/MIT_split/train", transform=transformation)
    data_test = ImageFolder("~/data/Master/MIT_split/test", transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(data_test, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, num_workers=cfg.num_workers)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model to be used
    model = WraperModel(num_classes=cfg.output_dim, feature_extraction=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    NUM_EPOCHS = cfg.num_epochs

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    # For early stopping
    stopper = EarlyStopping(patience = cfg.patience, min_delta = cfg.min_delta)
    
    for epoch in tqdm.tqdm(range(NUM_EPOCHS), desc="TRAINING THE MODEL"):
        _, _, train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        _, _, _, test_loss, test_accuracy = test(model, test_loader, criterion, device)

        # History
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Early stopping
        stopper(test_loss, model)
        if stopper.early_stop == True:
            print(f"Stopping training in epoch {epoch}")
            break
        
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
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
    
    print("Loading datasets...")
    data_train = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/train", transform=transformation)
    data_test = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/val", transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(data_test, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, num_workers=cfg.num_workers)
    
    # Get Input Dimensions
    C, H, W = np.asarray(data_train[0][0]).shape
    input_dim = C * H * W
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model_type == 'cnn':
        model = SimpleCNN(in_channels = C, hidden_channels=cfg.hidden_dim, output_d=cfg.output_dim, img_size= H * W)
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
    
    if cfg.task_type == "mlp_svm":
        svm_preds, _, svm_acc = train_svm(train_loader, test_loader, model, device)
        wandb.log({"svm_fold_acc": svm_acc})

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
    run_experiment()
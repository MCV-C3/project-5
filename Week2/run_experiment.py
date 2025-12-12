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
from sklearn.model_selection import KFold

from models import SimpleModel
from main import train, test, plot_computational_graph, plot_metrics
from metrics import MetricsComputer

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
    data_train = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/train", transform=transformation)
    data_test = ImageFolder("~/mcv/datasets/C3/2526/places_reduced/val", transform=transformation) 
    
    kfold = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)

    #train_loader = DataLoader(data_train, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, num_workers=cfg.num_workers)
    # test_loader = DataLoader(data_test, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, num_workers=cfg.num_workers)

    # Store aggregated OOF (Out Of Fold) results for final metrics
    oof_val_preds = []
    oof_val_labels = []
    oof_val_probs = []
    
    # Get Input Dimensions
    C, H, W = np.asarray(data_train[0][0]).shape
    input_dim = C * H * W
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Do a mapping so we can know to which fold corresponds each train sample
    indices_per_fold = []
    for fold_idx, (train_index, test_index) in enumerate(kfold.split(data_train)):
        indices_per_fold.append(test_index)
    
    # --- CROSS VALIDATION LOOP ---
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data_train)):
        print(f"\n--- FOLD {fold + 1}/{cfg.k_folds} ---")
        
        # 1. Create Subsets and Loaders for this fold
        train_sub = Subset(data_train, train_ids)
        val_sub = Subset(data_train, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True, 
                                  num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.num_workers, pin_memory=True)

        # 2. Re-initialize Model & Optimizer (Must be fresh for every fold)
        model = SimpleModel(input_d=input_dim, hidden_d=cfg.hidden_dim, output_d=cfg.output_dim)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 3. Training Loop for this Fold
        for epoch in tqdm.tqdm(range(cfg.num_epochs), desc=f"Fold {fold+1} Epochs"):
            
            # Train on fold's training set
            _, _, _, train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            
            # Validate on fold's validation set
            _, _, _, val_loss, val_acc = test(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Log per-fold basic metrics for each epoch
            wandb.log({
                f"fold_{fold}_train_loss": train_loss,
                f"fold_{fold}_train_acc": train_acc,
                f"fold_{fold}_val_loss": val_loss,
                f"fold_{fold}_val_acc": val_acc,
                "epoch": epoch
            })

        # 4. END OF FOLD: Collect predictions for final confusion matrix
        # Rerun test one last time or use the last epoch's results if they are sufficient        
        final_val_preds, final_val_labels, final_val_probs, _, _ = test(model, val_loader, criterion, device)
        oof_val_preds.extend(final_val_preds)
        oof_val_labels.extend(final_val_labels)
        oof_val_probs.extend(final_val_probs)
        
    ################################
    # COMPUTE MORE COMPLEX METRICS #
    ################################

    metrics = MetricsComputer(y_pred_train=oof_val_preds, y_pred_test=None, 
                                y_true_train=oof_val_labels, y_true_test=None,
                                probas_train=oof_val_probs, probas_test=None, 
                                indices_per_fold=indices_per_fold)
    
    # BASIC METRICS (PRECISION, RECALL, F1, ACCURACY, AUC)
    metrics.compute_all_metrics()
    metrics_dict = metrics.get_all_metrics_dict()
    wandb.log(metrics_dict)
    
    #CONFMAT
    confmat_fig = metrics.plot_confusion_matrix()
    wandb.log({"confusion_matrix": wandb.Image(confmat_fig)})
    plt.close(confmat_fig)
    
    # ROC CURVE
    roc_fig = metrics.plot_roc_curve()
    wandb.log({"ROC_curve": wandb.Image(roc_fig)})
    plt.close(roc_fig)
        
        
    ###########################################################################
    ##################### A PARTIR D'AQUÍ, CODI ANTIC #########################
    ###########################################################################
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleModel(input_d=C*H*W, hidden_d=cfg.hidden_dim, output_d=cfg.output_dim)
    #plot_computational_graph(model, input_size=(1, C*H*W))  # Batch size of 1, input_dim=10

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    NUM_EPOCHS = cfg.num_epochs

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for epoch in tqdm.tqdm(range(NUM_EPOCHS), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy")
    
    metrics = MetricsComputer(y_pred_train, y_pred_test, 
                                   labels_train, labels_test,
                                   y_probas_train, y_probas_test, indices_per_fold)
    
    # BASIC METRICS (PRECISION, RECALL, F1, ACCURACY, AUC)
    metrics.compute_all_metrics()
    metrics_dict = metrics.get_all_metrics_dict()
    wandb.log(metrics_dict)
    
    #CONFMAT
    confmat_fig = metrics.plot_confusion_matrix()
    wandb.log({"confusion_matrix": wandb.Image(confmat_fig)})
    plt.close(confmat_fig)
    
    # ROC CURVE
    roc_fig = metrics.plot_roc_curve()
    wandb.log({"ROC_curve": wandb.Image(roc_fig)})
    plt.close(roc_fig)
    '''
if __name__ == "__main__":
    run_experiment()
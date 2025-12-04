import wandb

from main import train, test, Dataset, SPLIT_PATH, get_descriptors
from bovw import BOVW
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

import matplotlib.pyplot as plt

# metrics 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

def run_experiment():

    config_defaults = {
        "detector_type": "SIFT",
        "cache_train":"/home/bernat/MCV/C3/project/project-5/Week1/cache_train_descriptor.pkl",
        "cache_test":"/home/bernat/MCV/C3/project/project-5/Week1/cache_test_descriptor.pkl",
        "codebook_size": 50,
        "dataset_path": SPLIT_PATH,
        "classifier_algorithm": "LogisticRegression",
        "classifier_kwargs": {},
    }

    # Init wandb
    wandb.init(
        project="C3-Week1-BOVW",
        entity="project-5",
        config=config_defaults,
    )

    cfg = wandb.config

    print(f"Starting experiment with config: {cfg}")
    
    print("Loading datasets...")
    data_train = Dataset(ImageFolder=SPLIT_PATH + "train")
    data_test = Dataset(ImageFolder=SPLIT_PATH + "test")

    if cfg.classifier_algorithm == 'LogisticRegression':
        classifier = LogisticRegression(class_weight="balanced", **cfg.classifier_kwargs)
    elif cfg.classifier_algorithm == 'SVM':
        classifier = SVC(class_weight="balanced", **cfg.classifier_kwargs)
    
    
    det_kwargs = {}
    if 'nfeatures' in cfg:
        det_kwargs['nfeatures'] = cfg.nfeatures

    cb_kwargs = {}
    
    bovw = BOVW(
        detector_type=cfg.detector_type,
        codebook_size=cfg.codebook_size,
        detector_kwargs=det_kwargs,
        codebook_kwargs=cb_kwargs
    )
    


    print("Training the model...")
    y_pred_train, y_probas_train, labels_train = train(dataset=data_train, bovw=bovw, 
                                                       classifier=classifier, 
                                                       cache_file=cfg.cache_train)

    train_acc = accuracy_score(y_true=labels_train, y_pred=y_pred_train)

    print("Accuracy on Phase[Train]:", train_acc)

    
    print("Evaluating the model...")
    y_pred_test, y_probas_test, labels_test = test(dataset=data_test, bovw=bovw, 
                                                   classifier=classifier,
                                                       cache_file=cfg.cache_test)

    test_acc = accuracy_score(y_true=labels_test, y_pred=y_pred_test)
    
    print("Accuracy on Phase[Test]:", test_acc)
    ###################################
    # METRICS CALCULATION
    ##################################
    
    classes = np.unique(labels_test)
    n_classes = len(classes)
    
    # ---- Precision / Recall / F1 ----
    precision_macro = precision_score(labels_test, y_pred_test, average="macro")
    recall_macro = recall_score(labels_test, y_pred_test, average="macro")
    f1_macro = f1_score(labels_test, y_pred_test, average="macro")

    precision_per_class = precision_score(labels_test, y_pred_test, average=None)
    recall_per_class = recall_score(labels_test, y_pred_test, average=None)
    f1_per_class = f1_score(labels_test, y_pred_test, average=None)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(labels_test, y_pred_test)
    fig_cm, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(fig_cm)})
    plt.close(fig_cm)

    # ---- ROC Curve + AUC ----
    y_test_bin = np.zeros((len(labels_test), n_classes))
    for i, c in enumerate(classes):
        y_test_bin[:, i] = (labels_test == c).astype(int)

    auc_per_class = {}
    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probas_test[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_probas_test[:, i])
        auc_per_class[str(c)] = float(auc)
        ax_roc.plot(fpr, tpr, label=f"Class {c} (AUC={auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (One-vs-Rest)")
    ax_roc.legend()

    wandb.log({"ROC_curve": wandb.Image(fig_roc)})
    plt.close(fig_roc)

    # ---- AUC aggregated ----
    auc_macro = roc_auc_score(y_test_bin, y_probas_test, average="macro", multi_class="ovr")
    auc_weighted = roc_auc_score(y_test_bin, y_probas_test, average="weighted", multi_class="ovr")

    
    
    wandb.log({
        # basic metrics
        "test_accuracy": float(test_acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),

        # per class
        "precision_per_class": {str(c): precision_per_class[i] for i, c in enumerate(classes)},
        "recall_per_class": {str(c): recall_per_class[i] for i, c in enumerate(classes)},
        "f1_per_class": {str(c): f1_per_class[i] for i, c in enumerate(classes)},

        # auc
        "AUC_macro": float(auc_macro),
        "AUC_weighted": float(auc_weighted),
        "AUC_per_class": auc_per_class,
    })

    print(f"Experiment completed. Train Acc: {train_acc}, Test Acc: {test_acc}")

    wandb.finish()

if __name__ == "__main__":
    run_experiment()




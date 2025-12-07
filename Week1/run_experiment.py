import wandb
import gc

from main import train, test, TrainDataset, TestDataset
from bovw import BOVW
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

import matplotlib.pyplot as plt

from metrics import MetricsComputer

def run_experiment(wandb_config=None, experiment_config=None):
    """
    Run the BOVW experiment with the given configurations.
    
    Args:
        wandb_config (dict): Configuration for wandb initialization containing:
            - project (str): wandb project name
            - entity (str): wandb entity name
        experiment_config (dict): Configuration for the experiment containing:
            - detector_type (str): Type of detector (SIFT, ORB, AKAZE, DenseSIFT)
            - detector_kwargs (dict): Keyword arguments for detector
            - codebook_kwargs (dict): Keyword arguments for codebook
            - pyramid_lvls (int): Number of pyramid levels
            - codebook_size (int): Size of codebook
            - normalize_histograms (bool): Whether to normalize histograms
            - use_pca (bool): Whether to use PCA
            - n_pca (int): Number of PCA components
            - classifier_algorithm (str): Classifier type (LogisticRegression or SVM)
            - classifier_kwargs (dict): Keyword arguments for classifier
    """
    
    # Default configurations
    default_wandb_config = {
        "project": "C3-Week1-BOVW",
        "entity": "project-5",
    }
    
    default_experiment_config = {
        "detector_type": "SIFT",
        "detector_kwargs": {},
        "codebook_kwargs": {},
        "pyramid_lvls": 1,
        "codebook_size": 50,
        "normalize_histograms": True,
        "use_pca": False,
        "n_pca": 64,
        "classifier_algorithm": "LogisticRegression",
        "classifier_kwargs": {},
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
    
    print("Loading datasets...")
    data_train = TrainDataset()
    data_test = TestDataset()

    if cfg.classifier_algorithm == 'LogisticRegression':
        classifier = LogisticRegression(class_weight="balanced", **dict(cfg.classifier_kwargs))
    elif cfg.classifier_algorithm == 'SVM':
        classifier = SVC(class_weight="balanced", probability=True, **dict(cfg.classifier_kwargs))
    
    bovw = BOVW(
        detector_type=cfg.detector_type,
        codebook_size=cfg.codebook_size,
        detector_kwargs=dict(cfg.detector_kwargs),
        codebook_kwargs=dict(cfg.codebook_kwargs),
        pyramid_lvls=cfg.pyramid_lvls,
        normalize=cfg.normalize_histograms,
        use_pca=cfg.use_pca,
        n_pca=cfg.n_pca,
        stride=cfg.stride,
        scale=cfg.scale
    )
    
    # Compute cache paths
    det_kwargs = dict(cfg.detector_kwargs)
    kwarg_detector_str = [f"_{str(key)}-{str(value)}" for key, value in det_kwargs.items()]
    kwarg_detector_str = "".join(kwarg_detector_str)
    cache_train = "/home/bernat/MCV/C3/project/project-5/cache_train/"
    cache_test = "/home/bernat/MCV/C3/project/project-5/cache_test/"
    cache_train = "./cache_train/"
    cache_test = "./cache_test/"
    cache_file_train = cache_train + cfg.detector_type + kwarg_detector_str+".pkl"
    cache_file_test = cache_test + cfg.detector_type + kwarg_detector_str+".pkl"
    
    print("Training the model...")
    
    y_pred_train, y_probas_train, labels_train = train(dataset=data_train, bovw=bovw, 
                                                       classifier=classifier, 
                                                       cache_file=cache_file_train)

    train_acc = accuracy_score(y_true=labels_train, y_pred=y_pred_train)

    print("Accuracy on Phase[Train]:", train_acc)

    
    print("Evaluating the model...")
    y_pred_test, y_probas_test, labels_test = test(dataset=data_test, bovw=bovw, 
                                                   classifier=classifier,
                                                   cache_file=cache_file_test)

    test_acc = accuracy_score(y_true=labels_test, y_pred=y_pred_test)
    
    print("Accuracy on Phase[Test]:", test_acc)
    ###################################
    # METRICS CALCULATION
    ##################################
    
    metrics = MetricsComputer(y_pred_train, y_pred_test, 
                                   labels_train, labels_test,
                                   y_probas_train, y_probas_test)
    
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
    

    
    
    
    # # ---- Precision / Recall / F1 ----
    # precision_macro = precision_score(labels_test, y_pred_test, average="macro")
    # recall_macro = recall_score(labels_test, y_pred_test, average="macro")
    # f1_macro = f1_score(labels_test, y_pred_test, average="macro")

    # precision_per_class = precision_score(labels_test, y_pred_test, average=None)
    # recall_per_class = recall_score(labels_test, y_pred_test, average=None)
    # f1_per_class = f1_score(labels_test, y_pred_test, average=None)

    # # ---- Confusion Matrix ----
    # cm = confusion_matrix(labels_test, y_pred_test)
    # fig_cm, ax = plt.subplots(figsize=(6, 6))
    # ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax)
    # ax.set_title("Confusion Matrix")
    # wandb.log({"confusion_matrix": wandb.Image(fig_cm)})
    # plt.close(fig_cm)

    # # ---- ROC Curve + AUC ----
    # y_test_bin = np.zeros((len(labels_test), n_classes))
    # for i, c in enumerate(classes):
    #     y_test_bin[:, i] = (labels_test == c).astype(int)

    # auc_per_class = {}
    # fig_roc, ax_roc = plt.subplots(figsize=(7, 7))

    # for i, c in enumerate(classes):
    #     fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probas_test[:, i])
    #     auc = roc_auc_score(y_test_bin[:, i], y_probas_test[:, i])
    #     auc_per_class[str(c)] = float(auc)
    #     ax_roc.plot(fpr, tpr, label=f"Class {c} (AUC={auc:.3f})")

    # ax_roc.plot([0, 1], [0, 1], "--")
    # ax_roc.set_xlabel("False Positive Rate")
    # ax_roc.set_ylabel("True Positive Rate")
    # ax_roc.set_title("ROC Curve (One-vs-Rest)")
    # ax_roc.legend()

    # wandb.log({"ROC_curve": wandb.Image(fig_roc)})
    # plt.close(fig_roc)

    # # ---- AUC aggregated ----
    # auc_macro = roc_auc_score(y_test_bin, y_probas_test, average="macro", multi_class="ovr")
    # auc_weighted = roc_auc_score(y_test_bin, y_probas_test, average="weighted", multi_class="ovr")

    
    
    # wandb.log({
    #     # basic metrics
    #     "test_accuracy": float(test_acc),
    #     "precision_macro": float(precision_macro),
    #     "recall_macro": float(recall_macro),
    #     "f1_macro": float(f1_macro),

    #     # per class
    #     "precision_per_class": {str(c): precision_per_class[i] for i, c in enumerate(classes)},
    #     "recall_per_class": {str(c): recall_per_class[i] for i, c in enumerate(classes)},
    #     "f1_per_class": {str(c): f1_per_class[i] for i, c in enumerate(classes)},

    #     # auc
    #     "AUC_macro": float(auc_macro),
    #     "AUC_weighted": float(auc_weighted),
    #     "AUC_per_class": auc_per_class,
    # })

    # DELETE VARIABLES TO FREE MEMORY
    del data_train, data_test
    del bovw, classifier
    del y_pred_train, y_probas_train, labels_train
    del y_pred_test, y_probas_test, labels_test

    plt.close('all')
    gc.collect()
    print(f"Experiment completed. Train Acc: {train_acc}, Test Acc: {test_acc}")

    wandb.finish()

if __name__ == "__main__":
    config_experiment = {
        "detector_type": "SIFT",
        "detector_kwargs": {},
        "codebook_kwargs": {},
        "pyramid_lvls": 1,
        "codebook_size": 512,
        "normalize_histograms": True,
        "use_pca": False,
        "n_pca": 64,
        "classifier_algorithm": "SVM",
        "classifier_kwargs": {
            "C": 1,
            "kernel": "rbf"
        }
    }
    config_wandb = {
        "project": "C3-Week1-BOVW",
        "entity": "project-5",
    }
    run_experiment(config_wandb, config_experiment)




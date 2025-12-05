import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)


class MetricsComputer:
    """
    A class to compute and store all evaluation metrics for classification models.
    
    Attributes:
        y_pred_train: Predicted labels for training set
        y_pred_test: Predicted labels for test set
        y_true_train: Ground truth labels for training set
        y_true_test: Ground truth labels for test set
        probas_train: Predicted probabilities for training set
        probas_test: Predicted probabilities for test set
    """
    
    def __init__(self, y_pred_train, y_pred_test, y_true_train, y_true_test, 
                 probas_train, probas_test):
        """
        Initialize the MetricsComputer.
        
        Args:
            y_pred_train: Predicted labels for training set
            y_pred_test: Predicted labels for test set
            y_true_train: Ground truth labels for training set
            y_true_test: Ground truth labels for test set
            probas_train: Predicted probabilities for training set
            probas_test: Predicted probabilities for test set
        """
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        self.y_true_train = y_true_train
        self.y_true_test = y_true_test
        self.probas_train = probas_train
        self.probas_test = probas_test
        
        self.classes = np.unique(y_true_test)
        self.n_classes = len(self.classes)
        
        # Initialize metric dictionaries
        self.train_metrics = {}
        self.test_metrics = {}
    
    def compute_accuracy(self):
        """Compute accuracy for train and test sets."""
        train_acc = accuracy_score(self.y_true_train, self.y_pred_train)
        test_acc = accuracy_score(self.y_true_test, self.y_pred_test)
        
        self.train_metrics['accuracy'] = train_acc
        self.test_metrics['accuracy'] = test_acc
        
        return train_acc, test_acc
    
    def compute_precision_recall_f1(self):
        """Compute precision, recall, and F1 scores (macro and per-class)."""
        # Train metrics
        train_precision_macro = precision_score(self.y_true_train, self.y_pred_train, average="macro", zero_division=0)
        train_recall_macro = recall_score(self.y_true_train, self.y_pred_train, average="macro", zero_division=0)
        train_f1_macro = f1_score(self.y_true_train, self.y_pred_train, average="macro", zero_division=0)
        
        train_precision_per_class = precision_score(self.y_true_train, self.y_pred_train, average=None, zero_division=0)
        train_recall_per_class = recall_score(self.y_true_train, self.y_pred_train, average=None, zero_division=0)
        train_f1_per_class = f1_score(self.y_true_train, self.y_pred_train, average=None, zero_division=0)
        
        # Test metrics
        test_precision_macro = precision_score(self.y_true_test, self.y_pred_test, average="macro", zero_division=0)
        test_recall_macro = recall_score(self.y_true_test, self.y_pred_test, average="macro", zero_division=0)
        test_f1_macro = f1_score(self.y_true_test, self.y_pred_test, average="macro", zero_division=0)
        
        test_precision_per_class = precision_score(self.y_true_test, self.y_pred_test, average=None, zero_division=0)
        test_recall_per_class = recall_score(self.y_true_test, self.y_pred_test, average=None, zero_division=0)
        test_f1_per_class = f1_score(self.y_true_test, self.y_pred_test, average=None, zero_division=0)
        
        # Store in dictionaries
        self.train_metrics['precision_macro'] = train_precision_macro
        self.train_metrics['recall_macro'] = train_recall_macro
        self.train_metrics['f1_macro'] = train_f1_macro
        self.train_metrics['precision_per_class'] = {str(c): train_precision_per_class[i] for i, c in enumerate(self.classes)}
        self.train_metrics['recall_per_class'] = {str(c): train_recall_per_class[i] for i, c in enumerate(self.classes)}
        self.train_metrics['f1_per_class'] = {str(c): train_f1_per_class[i] for i, c in enumerate(self.classes)}
        
        self.test_metrics['precision_macro'] = test_precision_macro
        self.test_metrics['recall_macro'] = test_recall_macro
        self.test_metrics['f1_macro'] = test_f1_macro
        self.test_metrics['precision_per_class'] = {str(c): test_precision_per_class[i] for i, c in enumerate(self.classes)}
        self.test_metrics['recall_per_class'] = {str(c): test_recall_per_class[i] for i, c in enumerate(self.classes)}
        self.test_metrics['f1_per_class'] = {str(c): test_f1_per_class[i] for i, c in enumerate(self.classes)}
    
    def compute_confusion_matrix(self):
        """Compute confusion matrix for test set."""
        cm = confusion_matrix(self.y_true_test, self.y_pred_test)
        self.test_metrics['confusion_matrix'] = cm
        return cm
    
    def plot_confusion_matrix(self):
        """Plot and return confusion matrix figure."""
        cm = self.compute_confusion_matrix()
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes).plot(ax=ax)
        ax.set_title("Confusion Matrix")
        return fig
    
    def compute_auc(self):
        """Compute AUC scores (per-class, macro, weighted)."""
        # Convert to binary format for AUC computation
        y_test_bin = np.zeros((len(self.y_true_test), self.n_classes))
        for i, c in enumerate(self.classes):
            y_test_bin[:, i] = (self.y_true_test == c).astype(int)
        
        y_train_bin = np.zeros((len(self.y_true_train), self.n_classes))
        for i, c in enumerate(self.classes):
            y_train_bin[:, i] = (self.y_true_train == c).astype(int)
        
        # Train AUC
        train_auc_macro = roc_auc_score(y_train_bin, self.probas_train, average="macro", multi_class="ovr")
        train_auc_weighted = roc_auc_score(y_train_bin, self.probas_train, average="weighted", multi_class="ovr")
        
        # Test AUC
        test_auc_macro = roc_auc_score(y_test_bin, self.probas_test, average="macro", multi_class="ovr")
        test_auc_weighted = roc_auc_score(y_test_bin, self.probas_test, average="weighted", multi_class="ovr")
        
        # Per-class AUC
        test_auc_per_class = {}
        for i, c in enumerate(self.classes):
            auc = roc_auc_score(y_test_bin[:, i], self.probas_test[:, i])
            test_auc_per_class[str(c)] = float(auc)
        
        self.train_metrics['AUC_macro'] = train_auc_macro
        self.train_metrics['AUC_weighted'] = train_auc_weighted
        self.test_metrics['AUC_macro'] = test_auc_macro
        self.test_metrics['AUC_weighted'] = test_auc_weighted
        self.test_metrics['AUC_per_class'] = test_auc_per_class
        
        return y_test_bin
    
    def plot_roc_curve(self, y_test_bin=None):
        """Plot ROC curve for all classes."""
        if y_test_bin is None:
            y_test_bin = self.compute_auc()
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        for i, c in enumerate(self.classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.probas_test[:, i])
            auc = roc_auc_score(y_test_bin[:, i], self.probas_test[:, i])
            ax.plot(fpr, tpr, label=f"Class {c} (AUC={auc:.3f})")
        
        ax.plot([0, 1], [0, 1], "--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (One-vs-Rest)")
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig
    
    def compute_all_metrics(self):
        """Compute all metrics at once."""
        self.compute_accuracy()
        self.compute_precision_recall_f1()
        self.compute_confusion_matrix()
        y_test_bin = self.compute_auc()
        return y_test_bin
    
    def get_test_metrics_dict(self):
        """Return test metrics as dictionary for logging."""
        return self.test_metrics
    
    def get_train_metrics_dict(self):
        """Return train metrics as dictionary for logging."""
        return self.train_metrics
    
    def get_all_metrics_dict(self):
        """Return all metrics (train + test) as dictionary."""
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def print_metrics(self):
        """Print all metrics to console."""
        print("\n" + "="*50)
        print("TRAINING METRICS")
        print("="*50)
        print(f"Accuracy: {self.train_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Precision (macro): {self.train_metrics.get('precision_macro', 'N/A'):.4f}")
        print(f"Recall (macro): {self.train_metrics.get('recall_macro', 'N/A'):.4f}")
        print(f"F1 Score (macro): {self.train_metrics.get('f1_macro', 'N/A'):.4f}")
        
        print("\n" + "="*50)
        print("TEST METRICS")
        print("="*50)
        print(f"Accuracy: {self.test_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Precision (macro): {self.test_metrics.get('precision_macro', 'N/A'):.4f}")
        print(f"Recall (macro): {self.test_metrics.get('recall_macro', 'N/A'):.4f}")
        print(f"F1 Score (macro): {self.test_metrics.get('f1_macro', 'N/A'):.4f}")
        print(f"AUC (macro): {self.test_metrics.get('AUC_macro', 'N/A'):.4f}")
        print(f"AUC (weighted): {self.test_metrics.get('AUC_weighted', 'N/A'):.4f}")
        print("="*50 + "\n")

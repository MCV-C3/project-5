import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

class BaseMetricsComputer:
    """
    Base class for computing core classification metrics.
    Handles data initialization and common metric computations (Accuracy, P/R/F1).
    """
    
    def __init__(self, y_true, y_pred, probas=None, classes=None):
        """
        Args:
            y_true: Ground truth labels (numpy array).
            y_pred: Predicted labels (numpy array).
            probas: Predicted probabilities (optional, numpy array).
            classes: List of class names/labels (optional).
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.probas = np.array(probas) if probas is not None else None
        
        # Determine classes automatically if not provided
        if classes is not None:
            self.classes = classes
        else:
            self.classes = np.unique(self.y_true)
            
        self.n_classes = len(self.classes)
        self.metrics = {}

    def compute_generic_metrics(self, prefix=""):
        """Computes Accuracy, Precision, Recall, and F1 (Macro & Per-Class)."""
        
        # Accuracy
        acc = accuracy_score(self.y_true, self.y_pred)
        self.metrics[f'{prefix}accuracy'] = acc
        
        # Macro Averages
        self.metrics[f'{prefix}precision_macro'] = precision_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.metrics[f'{prefix}recall_macro'] = recall_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.metrics[f'{prefix}f1_macro'] = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        
        # Per-class Metrics (stored as dicts)
        p_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        r_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        f_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        self.metrics[f'{prefix}precision_per_class'] = {str(c): p_class[i] for i, c in enumerate(self.classes)}
        self.metrics[f'{prefix}recall_per_class'] = {str(c): r_class[i] for i, c in enumerate(self.classes)}
        self.metrics[f'{prefix}f1_per_class'] = {str(c): f_class[i] for i, c in enumerate(self.classes)}
        
        return self.metrics

    def get_metrics(self):
        return self.metrics

class TrainMetrics(BaseMetricsComputer):
    """
    Subclass specifically for Training data. 
    Usually, we only need basic metrics to track convergence.
    """
    def __init__(self, y_true, y_pred, probas=None, classes=None):
        super().__init__(y_true, y_pred, probas, classes)

    def compute(self):
        """Compute standard training metrics."""
        return self.compute_generic_metrics(prefix="train_")


class TestMetrics(BaseMetricsComputer):
    """
    Subclass for Test/Validation data. 
    """
    def __init__(self, y_true, y_pred, probas=None, classes=None):
        super().__init__(y_true, y_pred, probas, classes)

    def compute(self):
        """Compute all test metrics including AUC and Confusion Matrix."""
        self.compute_generic_metrics(prefix="test_")
        self.compute_auc()
        self.compute_confusion_matrix()
        return self.metrics

    def compute_confusion_matrix(self):
        """Compute confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.metrics['test_confusion_matrix'] = cm
        return cm

    def compute_auc(self):
        """Compute AUC scores (Macro, Weighted, Per-Class)."""
        if self.probas is None:
            print("Warning: probabilities not provided. Skipping AUC computation.")
            return

        # Binarize labels for OvR AUC
        y_bin = np.zeros((len(self.y_true), self.n_classes))
        for i, c in enumerate(self.classes):
            y_bin[:, i] = (self.y_true == c).astype(int)

        try:
            self.metrics['test_AUC_macro'] = roc_auc_score(y_bin, self.probas, average="macro", multi_class="ovr")
            self.metrics['test_AUC_weighted'] = roc_auc_score(y_bin, self.probas, average="weighted", multi_class="ovr")
            
            # Per-class AUC
            auc_per_class = {}
            for i, c in enumerate(self.classes):
                auc_per_class[str(c)] = roc_auc_score(y_bin[:, i], self.probas[:, i])
            self.metrics['test_AUC_per_class'] = auc_per_class
            
        except ValueError as e:
            print(f"Skipping AUC due to error (likely missing classes in batch): {e}")

    def plot_confusion_matrix(self):
        """Returns a matplotlib figure of the confusion matrix."""
        if 'test_confusion_matrix' not in self.metrics:
            self.compute_confusion_matrix()
            
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(
            confusion_matrix=self.metrics['test_confusion_matrix'], 
            display_labels=self.classes
        ).plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix")
        return fig

    def plot_roc_curve(self):
        """Returns a matplotlib figure of the ROC curve."""
        if self.probas is None:
            return None
            
        y_bin = np.zeros((len(self.y_true), self.n_classes))
        for i, c in enumerate(self.classes):
            y_bin[:, i] = (self.y_true == c).astype(int)
            
        fig, ax = plt.subplots(figsize=(7, 7))
        for i, c in enumerate(self.classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], self.probas[:, i])
            auc_score = roc_auc_score(y_bin[:, i], self.probas[:, i])
            ax.plot(fpr, tpr, label=f"Class {c} (AUC={auc_score:.2f})")
        
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        return fig


class FoldMetrics:
    """
    Class specifically for handling Cross-Validation metrics.
    """
    def __init__(self, y_true_full, y_pred_full, indices_per_fold):
        """
        Args:
            y_true_full: The complete array of ground truth labels for the dataset.
            y_pred_full: The complete array of predicted labels.
            indices_per_fold: List of arrays, where each array contains the *validation* indices for a fold.
        """
        self.y_true = np.array(y_true_full)
        self.y_pred = np.array(y_pred_full)
        self.indices_per_fold = indices_per_fold
        self.fold_metrics = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': []
        }

    def compute(self):
        """Iterates through folds and computes metrics for the validation set of each fold."""
        
        for fold_idx, val_indices in enumerate(self.indices_per_fold):
            # Slice data for this fold
            y_true_fold = self.y_true[val_indices]
            y_pred_fold = self.y_pred[val_indices]
            
            # Compute core metrics for this fold
            self.fold_metrics['accuracy'].append(
                accuracy_score(y_true_fold, y_pred_fold)
            )
            self.fold_metrics['precision_macro'].append(
                precision_score(y_true_fold, y_pred_fold, average="macro", zero_division=0)
            )
            self.fold_metrics['recall_macro'].append(
                recall_score(y_true_fold, y_pred_fold, average="macro", zero_division=0)
            )
            self.fold_metrics['f1_macro'].append(
                f1_score(y_true_fold, y_pred_fold, average="macro", zero_division=0)
            )
            
        return self.fold_metrics

    def get_summary(self):
        """Returns mean and std of metrics across folds."""
        summary = {}
        for metric, values in self.fold_metrics.items():
            summary[f'fold_{metric}_mean'] = np.mean(values)
            summary[f'fold_{metric}_std'] = np.std(values)
        return summary
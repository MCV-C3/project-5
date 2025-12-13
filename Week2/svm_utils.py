import torch
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Extracting Features"):
                inputs = inputs.to(device)
                
                feats = model(inputs, return_features=True)
                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.numpy())

    return np.concatenate(features_list), np.concatenate(labels_list)


def train_svm(train_loader, val_loader, model, device):
    print("Extracting features for SVM...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_val, y_val = extract_features(model, val_loader, device)

    print("Training SVM...")
    clf  = SVC(kernel='rbf', C=1.0,  probability=True, random_state=42)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_acc = np.mean(val_preds == y_val)

    print(f"SVM Fold Accuracy: {val_acc:.4f}")

    return val_preds, y_val, val_acc
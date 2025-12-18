import torch
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.stats import mode
import gc
from fisher_vectors import FisherVectors


def extract_features_old(model, dataloader, device):
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

def extract_features(model, dataloader, device):
    model.eval()
    
    # 1. Determine feature dimension by doing a single forward pass
    print("Determine feature dimension")
    sample_input, _ = next(iter(dataloader))
    with torch.no_grad():
        sample_feat = model(sample_input.to(device), return_features=True)
    
    feat_dim = sample_feat.shape[1]
    total_samples = len(dataloader.dataset)
    
    # 2. Pre-allocate NumPy arrays (faster than list appends)
    print("preallocate np arrays")
    all_features = np.empty((total_samples, feat_dim), dtype=np.float32)
    all_labels = np.empty(total_samples, dtype=np.int64)
    
    pointer = 0
    with torch.no_grad():
        # Use non_blocking=True for faster host-to-device transfer
        for inputs, labels in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # Forward pass
            feats = model(inputs, return_features=True)
            
            # 3. Copy directly into the pre-allocated array
            all_features[pointer:pointer + batch_size] = feats.cpu().numpy()
            all_labels[pointer:pointer + batch_size] = labels.numpy()
            
            pointer += batch_size
    print("end extract features ")
    return all_features, all_labels


def train_svm(train_loader, val_loader, model, device, fisher=False, pca=False):
    
    print("Extracting features to train SVM...")
    X_train, y_train = extract_features(model, train_loader, device)
    gc.collect()
    torch.cuda.empty_cache() # Force GPU memory cleanup

    print("Extracting features to test SVM...")
    X_val, y_val = extract_features(model, val_loader, device)
    gc.collect() # Force CPU memory cleanup
    
    if pca:
        print("Fitting PCA")
        pca_dim = PCA(n_components=32)
        pca_dim.fit(X_train)
        X_train = pca_dim.transform(X_train)
        X_val = pca_dim.transform(X_val)

    if fisher:
        print("Calculating fisher vectors")
        fishing = FisherVectors(n_components=25)
        fishing.fit(X_train)
        
        X_train = fishing.compute_fisher_vectors(X_train)
        X_val = fishing.compute_fisher_vectors(X_val)
        
    print("Training SVM...")
    # clf  = SVC(kernel='rbf', C=1.0,  probability=True, random_state=42)
    clf  = LinearSVC(C=1.0, random_state=42)
    clf  = CalibratedClassifierCV(clf)
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict(X_val)
    val_acc = np.mean(val_preds == y_val)

    print(f"SVM Fold Accuracy: {val_acc:.4f}")

    return val_preds, y_val, val_acc

def voting_ensemble(preds_list):
    preds_array = np.array(preds_list)
    final_preds, _ = mode(preds_array, axis=0)
    return final_preds.flatten()

def sum_ensemble(preds_list):
    summed_preds = np.sum(np.array(preds_list), axis=0)
    final_preds = np.argmax(summed_preds, axis=1)
    return final_preds

def train_svm_patches(train_loader, val_loader, model, device, patches_lvl, method="vote", fisher=False, pca=False): #sum
    n_patches = 4**patches_lvl
    
    print("Extracting features to train SVM...")
    X_train, y_train = extract_features(model, train_loader, device)
    gc.collect()
    torch.cuda.empty_cache() # Force GPU memory cleanup
    print("Extracting features to test SVM...")
    X_val, y_val = extract_features(model, val_loader, device)
    gc.collect() # Force CPU memory cleanup
    
    
    
       
    if pca:
        print("Fitting PCA")
        pca_dim = PCA(n_components=32)
        pca_dim.fit(X_train)
        X_train = pca_dim.transform(X_train)
        X_val = pca_dim.transform(X_val)

    print(f"X_train before fisher: {X_train.shape}")
    print(f"X_val before fisher: {X_val.shape}") 
    if fisher:
        fishing = FisherVectors(n_components=25, covariance_type='diag', random_state=42, max_samples=None)
        fishing.fit(X_train)

        X_train = fishing.compute_fisher_vectors(X_train)
        X_val = fishing.compute_fisher_vectors(X_val)
    
    print(f"X_train after fisher: {X_train.shape}")
    print(f"X_val after fisher: {X_val.shape}")

    print("Training SVM...")
    # clf  = SVC(kernel='rbf', C=1.0,  probability=True, random_state=42)
    clf  = LinearSVC(C=1.0, random_state=42)
    clf  = CalibratedClassifierCV(clf)
    clf.fit(X_train, y_train)
    
    y_val_images = y_val[::n_patches]
    
    if method == "vote":
        patch_preds = clf.predict_proba(X_val)
        
        patch_preds_grouped = patch_preds.reshape(-1, n_patches)
        print(f"n_patches: {n_patches}")
        print(f"patch_preds_grouped shape (should be batch_size*n_patches): {patch_preds_grouped.shape}")
        
        final_preds, _ = mode(patch_preds_grouped, axis=1)
        final_preds = final_preds.flatten()
    elif method == "sum":
        patch_probs = clf.predict_proba(X_val)
        patch_probs_grouped = patch_probs.reshape(-1, n_patches, patch_probs.shape[1])
    
        summed_probs = np.sum(patch_probs_grouped, axis=1)
        
        final_preds = np.argmax(summed_probs, axis=1)
    
        
    val_acc_img = np.mean(final_preds == y_val_images)    
    


    return final_preds, y_val_images, val_acc_img
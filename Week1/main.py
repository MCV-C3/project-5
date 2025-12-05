from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

SPLIT_PATH = "../data/MIT_split/"


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"], kpts: Literal["N", "T"]):
    return np.array([bovw._compute_codebook_descriptor(kpts=kpt, descriptors=descriptor, kmeans=bovw.codebook_algo) for kpt,descriptor in zip(kpts,descriptors)])

def get_descriptors(dataset: List[Tuple[Type[Image.Image], int]], bovw: Type[BOVW], cache_file: str, split: str):

    # Try loading from cache to avoid recomputation
    if os.path.exists(cache_file):
    #if False:    
        print(f"Phase[{split}]: Loading descriptors from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
        except:
            raise ValueError(f"Could not load cache file {cache_file}")

        all_kpts = data['kpts']
        all_descriptors = data['descriptors']
        all_labels = data['labels']

    else:
        all_kpts = []
        all_descriptors = []
        all_labels = []

        for idx in tqdm.tqdm(range(len(dataset)), desc=f"Phase[{split}]: Extracting the descriptors"):

            image, label = dataset[idx]
            kpts, descriptors = bovw._extract_features(image=np.array(image))

            if descriptors is not None:
                all_kpts.append(kpts)
                all_descriptors.append(descriptors)
                all_labels.append(label)
        
        print(f"Saving features to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'kpts': all_kpts,
                'descriptors': all_descriptors,
                'labels': all_labels
            }, f)

    return all_kpts, all_descriptors, all_labels


def test(dataset: List[Tuple[Type[Image.Image], int]], bovw: Type[BOVW], 
          classifier: Type[object], cache_file: str=None):
    
    test_kpts, test_descriptors, descriptors_labels = get_descriptors(dataset, bovw, cache_file, split="Test")
    
    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(kpts=test_kpts, descriptors=test_descriptors, bovw=bovw)

    print("Predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    y_probas = classifier.predict_proba(bovw_histograms)

    return y_pred, y_probas, descriptors_labels 

def train(dataset: List[Tuple[Type[Image.Image], int]], bovw:Type[BOVW], 
          classifier: Type[object], k_folds: int=5, cache_file: str=None):

    all_kpts, all_descriptors, all_labels = get_descriptors(dataset, bovw, cache_file, split="Train")

    print("Fitting the codebook", end=" ")
    dt = bovw._update_fit_codebook(descriptors=all_descriptors)
    print(f"(took {dt} seconds)")

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(kpts=all_kpts, descriptors=all_descriptors, bovw=bovw) 
    
    print("Fitting the classifier")
    
    # Obtain the predictions and probabilities of the train set using cross-validation
    y_pred = cross_val_predict(
        estimator=classifier,
        X=bovw_histograms,
        y=all_labels,
        cv=k_folds,
        n_jobs=-1
    )
    y_probas = cross_val_predict(
        estimator=classifier,
        X=bovw_histograms, 
        y=all_labels, 
        cv=k_folds, 
        method='predict_proba',
        n_jobs=-1
    )
    
    classifier.fit(bovw_histograms, all_labels)
    #y_pred = classifier.predict(bovw_histograms)
    #y_probas = classifier.predict_proba(bovw_histograms)
    
    return y_pred, y_probas, all_labels


def Dataset(ImageFolder:str = SPLIT_PATH + "train") -> List[Tuple[Type[Image.Image], int]]:

    """
    Expected Structure:

        ImageFolder/<cls label>/xxx1.jpg
        ImageFolder/<cls label>/xxx2.jpg
        ImageFolder/<cls label>/xxx3.jpg
        ...

        Example:
            ImageFolder/cat/123.jpg
            ImageFolder/cat/nsdf3.jpg
            ImageFolder/cat/[...]/asd932_.jpg
    
    """

    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    
    dataset :List[Tuple] = []

    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):

        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")

            dataset.append((img_pil, map_classes[cls_folder]))


    return dataset

def TrainDataset() -> List[Tuple[Type[Image.Image], int]]:
    """Wrapper for training dataset"""
    return Dataset(ImageFolder=SPLIT_PATH+"train")

def TestDataset() -> List[Tuple[Type[Image.Image], int]]:
    """Wrapper for testing dataset"""
    return Dataset(ImageFolder=SPLIT_PATH+"test")


if __name__ == "__main__":
    data_train = TrainDataset()
    data_test = TestDataset()

    bovw = BOVW()
    classifier = LogisticRegression(class_weight="balanced")
    
    y_pred_train, _, all_labels = train(dataset=data_train, bovw=bovw, classifier=classifier, cache_file="D-SIFT_train_cache.pkl")
    
    acc_train = accuracy_score(y_true=all_labels, y_pred=y_pred_train)
    print(f"Accuracy on Phase[Train]: {acc_train:.4f}")

    y_pred_test, y_probas, descriptors_labels = test(dataset=data_test, bovw=bovw, classifier=classifier, cache_file="D-SIFT_test_cache.pkl")

    acc_test = accuracy_score(y_true=descriptors_labels, y_pred=y_pred_test)
    print(f"Accuracy on Phase[Test]: {acc_test:.4f}")
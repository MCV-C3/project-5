from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SPLIT_PATH = "/home/bernat/MCV/C3/project/project-5/data/places_reduced/"
CODEBOOK_SIZE = 100
DESCRIPTOR = "DENSE_SIFT"

# BATCH CONFIG
BATCH_PROCESSING = False
BATCH_SIZE = 512

# SIFT CONFIG
STEP_SIZE = 8 # must be at least 8 to match precomputed. Then must be multiple of 8
FEATURE_SCALE = 16
PRECOMPUTED = False


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])


def test(dataset: List[Tuple[Type[Image.Image], int]]
         , bovw: Type[BOVW], 
         classifier:Type[object]):
    
    test_descriptors = []
    descriptors_labels = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]: Extracting the descriptors"):
        image, label, precomputed_sift = dataset[idx]
        if precomputed_sift is not None:
            descriptors = np.load(precomputed_sift)
        else:
            _, descriptors = bovw._extract_features(image=np.array(image), 
                                                step_size=STEP_SIZE,
                                                feature_scale=FEATURE_SCALE)
        
        if descriptors is not None:
            test_descriptors.append(descriptors)
            descriptors_labels.append(label)
            
    
    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=test_descriptors, bovw=bovw)
    
    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    print("Accuracy on Phase[Test]:", accuracy_score(y_true=descriptors_labels, y_pred=y_pred))
    

def train(dataset: List[Tuple[Type[Image.Image], int]],
           bovw:Type[BOVW]):
    all_descriptors = []
    all_labels = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
        
        image, label, precomputed_sift = dataset[idx]
        if precomputed_sift is not None:
            descriptors = np.load(precomputed_sift)
        else:
            _, descriptors = bovw._extract_features(image=np.array(image), 
                                                step_size=STEP_SIZE,
                                                feature_scale=FEATURE_SCALE)
        
        if descriptors  is not None:
            all_descriptors.append(descriptors)
            all_labels.append(label)
            
    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, bovw=bovw) 
    
    print("Fitting the classifier")
    classifier = LogisticRegression(class_weight="balanced").fit(bovw_histograms, all_labels)

    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=classifier.predict(bovw_histograms)))
    
    return bovw, classifier

def batch_generator(dataset, batch_size):
    """Yields successive n-sized chunks from dataset."""
    for i in range(0, len(dataset), batch_size):
        images, labels, descriptor_paths = dataset[i:i + batch_size]
        descriptors = [np.load(descriptor_path)[:,:,STEP_SIZE//8] for descriptor_path in descriptor_paths]
        
        yield images, labels, descriptors

def train_batched(dataset: List[Tuple[Type[Image.Image], int]], 
                  bovw: Type[BOVW], 
                  batch_size: int = 64):
    print(f"Phase 1: Learning Codebook (Batch Size: {batch_size})")
    
    loader = batch_generator(dataset, batch_size)
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch in tqdm.tqdm(loader, total=total_batches, desc="Updating Codebook"):
        batch_descriptors = []
        
        for image, _, descriptors in batch:
            if descriptors is not None:
                batch_descriptors.append(descriptors)
                continue
            _, descriptors = bovw._extract_features(image=np.array(image))
            batch_descriptors.append(descriptors)
        
        if len(batch_descriptors) > 0:
            bovw._update_fit_codebook(descriptors=batch_descriptors)

    print("Codebook learning complete.")

    print("Phase 2: Generating Histograms for Classifier")
    
    bovw_histograms = []
    all_labels = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Encoding Images"):
        image, label = dataset[idx]
        
        _, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors is not None:
            hist = bovw._compute_codebook_descriptor(descriptors=descriptors, kmeans=bovw.codebook_algo)
            
            bovw_histograms.append(hist)
            all_labels.append(label)

    bovw_histograms = np.array(bovw_histograms)
    all_labels = np.array(all_labels)

    print(f"Fitting Classifier on {len(bovw_histograms)} samples...")
    
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(bovw_histograms, all_labels)

    y_pred = classifier.predict(bovw_histograms)
    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=y_pred))
    
    return bovw, classifier

def Dataset(ImageFolder:str = SPLIT_PATH + "train", precomputed:bool = False) -> List[Tuple[Type[Image.Image], int]]:

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
            
            if precomputed:
                precomputed_sift = os.path.join(image_path, "precomputed_sift", os.path.basename(img).replace(".jpg", ".npy"))
                dataset.append((img_pil, map_classes[cls_folder], precomputed_sift))
            else:
                dataset.append((img_pil, map_classes[cls_folder], None))


    return dataset


    


if __name__ == "__main__":
    data_train = Dataset(ImageFolder=SPLIT_PATH+"train", precomputed=PRECOMPUTED)
    data_test = Dataset(ImageFolder=SPLIT_PATH+"val", precomputed=PRECOMPUTED) 

    bovw = BOVW(detector_type=DESCRIPTOR,
                codebook_size=CODEBOOK_SIZE,
                detector_kwargs={},
                codebook_kwargs={'batch_size': 1024*4, 'random_state': 42})
    if not BATCH_PROCESSING:
        bovw, classifier = train(dataset=data_train, bovw=bovw)
    else:
        bovw, classifier = train_batched(dataset=data_train, bovw=bovw, batch_size=BATCH_SIZE)
    
    test(dataset=data_test, bovw=bovw, classifier=classifier)
    
    print("params used:")
    print(f" - Descriptor: {DESCRIPTOR}")
    print(f" - Codebook Size: {CODEBOOK_SIZE}")
    
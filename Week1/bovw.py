import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import time

from typing import *

IMGS_SHAPE = (256, 256)

class EmptyTransform():
    def fit_transform(self, X):
        return X
    
    def transform(self, X):
        return X

class BOVW():

    def __init__(self, detector_type="DenseSIFT", codebook_size:int=500, detector_kwargs:dict={}, codebook_kwargs:dict={}, pyramid_lvls:int=1, normalize:bool=True, use_pca:bool=False, n_pca:int=64, stride = 8, scale = 2):

        self.detector_type = detector_type

        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        elif detector_type == 'DenseSIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)

            # Define keypoint extractor for dense sampling
            self.kpts_extractor = (lambda im: [cv2.KeyPoint(x, y, scale) for y in range(0, im.shape[0], stride) for x in range(0, im.shape[1], stride)])
        else:
            raise ValueError("Detector type must be 'SIFT', 'SURF', or 'ORB'")
        
        self.codebook_size = codebook_size
        self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs, random_state=42)
        self.pyramid_lvls = pyramid_lvls
        self.normalize = normalize
        self.reduction_algo = PCA(n_components=n_pca) if use_pca else EmptyTransform()
        
               
    ## Modified to create a dense sift if needed
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        if self.detector_type == 'DenseSIFT':
            kpts, desc = self.detector.compute(image, self.kpts_extractor(image))
        else:
            kpts, desc = self.detector.detectAndCompute(image, None)
    
        # Kpts to NumPy array as (x,y)
        return np.array([kp.pt for kp in kpts]), desc
    
    
    def _update_fit_codebook(self, descriptors: Literal["N", "T", "d"]) -> str:
        
        all_descriptors = np.vstack(descriptors)

        # Dimensionality reduction before clustering
        all_descriptors = self.reduction_algo.fit_transform(all_descriptors)

        # K-Means clustering for codebook
        start = time.time()
        self.codebook_algo.partial_fit(X=all_descriptors)
        end = time.time()
        return f"{end - start:.2f}"
    

    def _compute_codebook_descriptor(self, kpts: Literal["1 T"], descriptors: Literal["1 T d"], kmeans: Type[KMeans]) -> np.ndarray:
        # Dimenstionality reduction before clustering
        descriptors = self.reduction_algo.transform(descriptors)

        visual_words = kmeans.predict(descriptors)

        # Kpts coordinates
        x_coords = kpts[:, 0]
        y_coords = kpts[:, 1]

        # Create descriptor with levels: 0 (1x1), 1 (2x2), 2 (4x4)...
        pyramid_descriptor = []
        for l in range(self.pyramid_lvls):
            divs = 2**l
            step_x = IMGS_SHAPE[1] / divs
            step_y = IMGS_SHAPE[0] / divs

            # Iterate through cells
            for i in range(divs):      # Y-axis
                for j in range(divs):  # X-axis

                    # Create boolean mask for points inside this cell
                    mask = (
                        (x_coords >= j*step_x) & (x_coords < (j+1)*step_x) & 
                        (y_coords >= i*step_y) & (y_coords < (i+1)*step_y)
                    )

                    # Get visual words belonging to this cell
                    cell_words = visual_words[mask]

                    # Create histogram of visual words
                    codebook_descriptor = np.zeros(kmeans.n_clusters)
                    for label in cell_words:
                        codebook_descriptor[label] += 1
                    
                    # Append to pyramid descriptor
                    pyramid_descriptor.append(codebook_descriptor)

        final_descriptor = np.concatenate(pyramid_descriptor)

        # Normalize the histograms (optional)
        if self.normalize:
            final_descriptor /= (np.linalg.norm(final_descriptor) + 1e-7)

        return final_descriptor


def visualize_bow_histogram(histogram, image_index, output_folder="./test_example.jpg"):
    """
    Visualizes the Bag of Visual Words histogram for a specific image and saves the plot to the output folder.
    
    Args:
        histogram (np.array): BoVW histogram.
        cluster_centers (np.array): Cluster centers (visual words).
        image_index (int): Index of the image for reference.
        output_folder (str): Folder where the plot will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram)
    plt.title(f"BoVW Histogram for Image {image_index}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.xticks(range(len(histogram)))
    
    # Save the plot to the output folder
    plot_path = os.path.join(output_folder, f"bovw_histogram_image_{image_index}.png")
    plt.savefig(plot_path)
    
    # Optionally, close the plot to free up memory
    plt.close()

    print(f"Plot saved to: {plot_path}")


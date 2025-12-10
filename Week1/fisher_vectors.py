import numpy as np
from sklearn.mixture import GaussianMixture
import time

class FisherVectors:
    def __init__(self, n_components:int=64, covariance_type:str='diag', random_state:int=42, 
                 max_samples:int=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_samples = max_samples
        self.gmm = None
        
    def fit(self, X):
        print(f"[Fisher] Stacking descriptors...")
        all_descriptors = np.vstack(X)
        print(f"[Fisher] Total descriptors available: {len(all_descriptors):,}")
        
        if self.max_samples is not None and len(all_descriptors) > self.max_samples:
            print(f"[Fisher] Subsampling to {self.max_samples:,} for GMM fitting...")
            np.random.seed(self.random_state)
            indices = np.random.choice(len(all_descriptors), self.max_samples, replace=False)
            data_to_fit = all_descriptors[indices]
        else:
            data_to_fit = all_descriptors

        print(f"[Fisher] Fitting sklearn GMM with {self.n_components} components (dim={data_to_fit.shape[1]})...")
        start = time.time()
        
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=100,
            verbose=1
        )
        self.gmm.fit(data_to_fit)
        
        elapsed = time.time() - start
        print(f"[Fisher] ✓ GMM fitted in {elapsed:.2f}s")

    def compute_fisher_vectors(self, descriptors: np.ndarray) -> np.ndarray:
        if self.gmm is None:
            raise ValueError("Model must be fitted first")
            
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(2 * self.n_components * descriptors.shape[1])

        means = self.gmm.means_
        covars = self.gmm.covariances_
        weights = self.gmm.weights_
        
        probs = self.gmm.predict_proba(descriptors)
        
        n_descriptors = descriptors.shape[0]
        n_components = self.n_components
        dim = descriptors.shape[1]
        
        sigma = np.sqrt(covars)
        sigma_inv = 1.0 / sigma
        
        fv_means = np.zeros((n_components, dim))
        fv_covars = np.zeros((n_components, dim))
        
        for k in range(n_components):
            p_k = probs[:, k].reshape(-1, 1)
            diff = descriptors - means[k]
            
            mult = p_k * diff
            fv_means[k] = np.sum(mult, axis=0) / sigma[k]
            
            mult_cov = p_k * ((diff**2) / (covars[k]) - 1)
            fv_covars[k] = np.sum(mult_cov, axis=0) 
            
        fv = np.concatenate([fv_means.flatten(), fv_covars.flatten()])
        fv /= np.sqrt(n_descriptors)
        
        return fv
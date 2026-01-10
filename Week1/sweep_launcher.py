import wandb
from run_experiment import run_experiment

# Wandb configuration
wandb_config = {
    "project": "C3-Week1-BOVW",
    "entity": "project-5",
}

# Layout for perfoming the sweeps for the different experiments
sweep_config = {
    'method': 'grid',
    'parameters': {
        'detector_type': {
            'values': ['DenseSIFT']
        },
        'detector_kwargs':{
            'values': [{}]  
        },
        'codebook_kwargs':{
            'values': [{}]  
        },
        'pyramid_lvls':{
            'values': [3]    
        },
        'codebook_size': {
            'values': [256]
        },
        'normalize_histograms':{
            'values': [True]    
        },
        'use_standard_scaling':{
            'values': [False]  
        },
        'use_pca':{
            'values': [False]    
        },
        'n_pca':{
            'values': [64]    
        },
        'stride':{
            'values': [8]
        },
        'scale':{
            'values': [8]
        },
        'classifier_algorithm':{
            'values': ['SVM']
        },
        'classifier_kwargs':{
            'values': [
                {'C': 1, 'kernel':'rbf'},
                ]
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}


sweep_config_test_detector_codebook = {
    'method': 'grid',
    'parameters': {
        # Compare different feature extractors
        'detector_type': {
            'values': ['SIFT', 'ORB', 'DenseSIFT'] 
        },
        'detector_kwargs': {
            'values': [{}] 
        },
        # Test the impact of vocabulary size
        'codebook_size': {
            'values': [256, 512, 1024]
        },
        # Keep these fixed for now to isolate feature performance
        'pyramid_lvls': {'values': [1]},
        'normalize_histograms': {'values': [True]},
        'use_pca': {'values': [False]},
        'n_pca':{'values': [64]},
        'stride':{'values': [8]},
        'scale':{'values': [2]},
        # 'classifier_algorithm': {'values': ['SVM']},
        # 'classifier_kwargs': {'values': [{'C': 1, 'kernel': 'rbf'}]},
        'classifier_algorithm': {'values': ['LogisticRegression']},
        'classifier_kwargs': {'values': [{}]}
    },
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'}
}

sweep_config_tune_dense_sift_params = {
    'method': 'grid',
    'parameters': {
        # Fixed Optimal Parameters (Assuming 512 was the best size)
        'codebook_size': {
            'values': [256] 
        },
        # Fixed Detector and Classifier
        'detector_type': {
            'values': ['DenseSIFT'] 
        },
        'classifier_algorithm': {
            'values': ['LogisticRegression']
        },
        'classifier_kwargs': {
            'values': [{}]
        },
        
        # Variable: Tune Stride and Scale
        'stride': {
            'values': [8, 16] # Explore finer (4) and coarser (16) grids
        },
        'scale': {
            'values': [2, 4, 8, 16, 32] # Explore smaller (2) and larger (32) feature patches
        },
        
        # Fixed Pipeline Constants
        'pyramid_lvls': {'values': [1]},
        'normalize_histograms': {'values': [True]},
        'use_pca': {'values': [False]},
        'n_pca': {'values': [64]},
        'detector_kwargs': {'values': [{}]}, 
    },
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'}
}


# Sweep configuration for Fisher Vectors with DenseSIFT
sweep_config_fisher_densesift = {
    'method': 'grid',
    'parameters': {
        # Fixed: DenseSIFT with stride=8, scale=8
        'detector_type': {
            'values': ['DenseSIFT']
        },
        'detector_kwargs': {
            'values': [{}]
        },
        'stride': {
            'values': [8]
        },
        'scale': {
            'values': [8]
        },
        
        # Fisher Vector encoding
        'encoding': {
            'values': ['fisher']
        },
        
        # Test different numbers of GMM components
        'n_components': {
            'values': [5, 10, 25, 40]
        },
        
        # Use PCA for dimensionality reduction
        'use_pca': {
            'values': [True]
        },
        'n_pca': {
            'values': [32]
        },
        
        # Fixed classifier: SVM with RBF kernel, C=1
        'classifier_algorithm': {
            'values': ['SVM']
        },
        'classifier_kwargs': {
            'values': [{'C': 1, 'kernel': 'rbf'}]
        },
        
        # Other parameters (not used for Fisher but needed)
        'codebook_size': {'values': [None]},  # Not used for Fisher
        'pyramid_lvls': {'values': [1]},
        'normalize_histograms': {'values': [True]},
        'codebook_kwargs': {'values': [{}]}
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    }
}

def run_experiment_with_wandb_config():
    """Wrapper function to pass wandb_config to run_experiment."""
    run_experiment(wandb_config=wandb_config)

sweep_id = wandb.sweep(sweep_config, project=wandb_config["project"], entity=wandb_config["entity"])
print(f"Initiated sweep with ID: {sweep_id}")
wandb.agent(sweep_id, function=run_experiment)

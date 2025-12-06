import wandb
from run_experiment import run_experiment

# Wandb configuration
wandb_config = {
    "project": "C3-Week1-BOVW",
    "entity": "project-5",
}

sweep_config = {
    'method': 'grid',
    'parameters': {
        'detector_type': {
            'values': ['SIFT', 'ORB', 'AKAZE', 'DenseSIFT']
        },
        'detector_kwargs':{
            'values': [
                #{'nfeatures': 10}
                {}
            ]  
        },
        'codebook_kwargs':{
            'values': [{}]  
        },
        'pyramid_lvls':{
            'values': [1]    
        },
        'codebook_size': {
            'values': [200]
        },
        'normalize_histograms':{
            'values': [True]    
        },
        'use_pca':{
            'values': [False]    
        },
        'n_pca':{
            'values': [64]    
        },
        'classifier_algorithm':{
            'values': ['SVM']
        },
        'classifier_kwargs':{
            'values': [
                {'C': 1,
                'kernel': 'rbf'}
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
            'values': [512, 1024]
        },
        # Keep these fixed for now to isolate feature performance
        'pyramid_lvls': {'values': [1]},
        'normalize_histograms': {'values': [True]},
        'use_pca': {'values': [False]},
        # 'classifier_algorithm': {'values': ['SVM']},
        # 'classifier_kwargs': {'values': [{'C': 1, 'kernel': 'rbf'}]},
        'classifier_algorithm': {'values': ['LogisticRegression']},
        'classifier_kwargs': {'values': [{}]}
    },
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'}
}

def run_experiment_with_wandb_config():
    """Wrapper function to pass wandb_config to run_experiment."""
    run_experiment(wandb_config=wandb_config)

sweep_id = wandb.sweep(sweep_config_test_detector_codebook, project=wandb_config["project"], entity=wandb_config["entity"])
print(f"Initiated sweep with ID: {sweep_id}")
wandb.agent(sweep_id, function=run_experiment)

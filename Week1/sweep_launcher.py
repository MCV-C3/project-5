import wandb
from run_experiment import run_experiment

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
            'values': [50]
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
                'kernel': 'linear'}
            ]
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

sweep_id = wandb.sweep(sweep_config, project="C3-Week1-BOVW", entity="project-5")
print(f"Initiated sweep with ID: {sweep_id}")
wandb.agent(sweep_id, function=run_experiment)

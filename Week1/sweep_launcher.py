import wandb
from run_experiment import run_experiment

sweep_config = {
    'method': 'grid',
    'parameters': {
        'detector_type': {
            'values': ['SIFT', 'ORB']
        },
        'codebook_size': {
            'values': [10, 20, 30]
        },
        'nfeatures': {
            'values': [10]
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

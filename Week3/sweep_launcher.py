import wandb
#from run_experiment.run_experiment_cross_val import run_experiment
from run_experiment.run_experiment_test import run_experiment

# Wandb configuration
wandb_config = {
    "project": "C3-Week3-MobileNet",
    "entity": "project-5",
}

# Layout for perfoming the sweeps for the different experiments
baseline_experiment = {
    'method': 'grid',
    'parameters': {
        'image_size': {
            'values': [(224,224)]
        },
        'batch_size': {
            'values': [256]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'output_dim': {
            'values': [8]
        },
        'num_epochs': {
            'values': [20]
        }, 
        'num_workers': {
            'values': [8]
        },
        'patience':{
            'values':[5]
        },
        'min_delta':{
            'values':[0]
        },
        'save_weights':{
            'values':[False]
        },
        'k_folds':{
            'values': [4]
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

def run_experiment_with_wandb_config():
    """Wrapper function to pass wandb_config to run_experiment."""
    run_experiment(wandb_config=wandb_config)

sweep_id = wandb.sweep(baseline_experiment, project=wandb_config["project"], entity=wandb_config["entity"])
print(f"Initiated sweep with ID: {sweep_id}")
wandb.agent(sweep_id, function=run_experiment)
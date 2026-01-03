import wandb
from run_experiment.run_experiment_cross_val import run_experiment
#from run_experiment.run_experiment_test import run_experiment

# Wandb configuration
wandb_config = {
    "project": "C3-Week3-MobileNet",
    "entity": "marc-org",
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
            'values':[True]
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

blocks_experiment = {
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
            'values': [50]
        },
        'feature_extraction':{
            'values': [True]
        },
        'blocks_to_keep':{
            'values': [list(range(17)),list(range(16)),list(range(15)),list(range(14)),list(range(13))]
        },
        'out_feat':{
            'values': [-1]
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
        },
        'aug_horizontal_flip':{
            'values':[True]
        },
        'aug_rotation':{
            'values':[True]
        },
        'aug_color_jitter':{
            'values':[True]
        },
        'aug_zoom':{
            'values':[True]
        },
        'aug_gaussian_blur':{
            'values':[True]
        },
        'use_imagenet_norm':{
            'values':[False]
        },
        'add_aug':{
            'values':[True]
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

train_all_experiment = {
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
        'feature_extraction':{
            'values': [False]
        },
        'blocks_to_keep':{
            'values': [list(range(14))]
        },
        'out_feat':{
            'values': [-1]
        },
        'num_workers': {
            'values': [8]
        },
        'patience':{
            'values':[20]
        },
        'min_delta':{
            'values':[0.001]
        },
        'save_weights':{
            'values':[False]
        },
        'k_folds':{
            'values': [4]
        },
        'aug_horizontal_flip':{
            'values':[True]
        },
        'aug_rotation':{
            'values':[True]
        },
        'aug_color_jitter':{
            'values':[True]
        },
        'aug_zoom':{
            'values':[True]
        },
        'aug_gaussian_blur':{
            'values':[True]
        },
        'use_imagenet_norm':{
            'values':[False]
        },
        'add_aug':{
            'values':[True]
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

sweep_id = wandb.sweep(train_all_experiment, project=wandb_config["project"], entity=wandb_config["entity"])
print(f"Initiated sweep with ID: {sweep_id}")
wandb.agent(sweep_id, function=run_experiment)
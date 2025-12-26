import wandb
from run_experiment_cross_val import run_experiment
# from run_experiment_test import run_experiment

# Wandb configuration
wandb_config = {
    "project": "C3-Week2-MLP",
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
        'hidden_dim': {
            'values': [300]
        },
        'output_dim': {
            'values': [11]
        },
        'num_epochs': {
            'values': [20]
        }, 
        'num_workers': {
            'values': [8]
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

image_size = {
    'method': 'grid',
    'parameters': {
        'image_size': {
            'values': [(2,2), (4,4), (8,8), (16, 16), (32, 32), (64, 64), (128, 128), (224, 224), (256, 256)]
        },
        'batch_size': {
            'values': [256]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'hidden_dim': {
            'values': [300]
        },
        'output_dim': {
            'values': [11]
        },
        'num_epochs': {
            'values': [20]
        }, 
        'num_workers': {
            'values': [8]
        },
        'task_type': {
            'values': ['mlp_only']
        },
        'model_type':{
            'values':['mlp']
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

svm_mlp = {
    'method': 'grid',
    'parameters': {
        'image_size': {
            'values': [(4,4)]
        },
        'batch_size': {
            'values': [256]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'hidden_dim': {
            'values': [300]
        },
        'output_dim': {
            'values': [11]
        },
        'num_epochs': {
            'values': [20]
        }, 
        'num_workers': {
            'values': [8]
        },
        'task_type': {
            'values': ['mlp_svm']
        },
        'model_type':{
            'values':['mlp']
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}


hidd_layers = {
    'method': 'grid',
    'parameters': {
        'image_size': {
            'values': [(4,4)]
        },
        'batch_size': {
            'values': [256]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'hidden_dim': {
            'values': [300]
        },
        'num_hidden_layers': {
            'values': [1, 2, 3, 4]
        },
        'output_dim': {
            'values': [11]
        },
        'num_epochs': {
            'values': [20]
        }, 
        'num_workers': {
            'values': [8]
        },
        'task_type': {
            'values': ['mlp_only']
        },
        'model_type':{
            'values':['mlp']
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

svm_mlp_patches = {
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
        'hidden_dim': {
            'values': [300]
        },
        'output_dim': {
            'values': [11]
        },
        'num_epochs': {
            'values': [20]
        }, 
        'num_workers': {
            'values': [8]
        },
        'task_type': {
            'values': ['mlp_svm']
        },
        'model_type':{
            'values':['mlp']
        },
        'patches_lvl':{
            'values':[0,1,2,3]
        },
        'patches_method':{
            'values':['sum']
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

cnn = {
    'method': 'grid',
    'parameters': {
        'image_size': {
            'values': [(256,256)]
        },
        'batch_size': {
            'values': [256]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'hidden_dim': {
            'values': [32]
        },
        'output_dim': {
            'values': [11]
        },
        'num_epochs': {
            'values': [20]
        }, 
        'num_workers': {
            'values': [8]
        },
        'task_type': {
            'values': ['mlp_only']
        },
        'model_type':{
            'values':['cnn']
        },
        'conv_stride': {
            'values': [1, 2, 3]
        },
        'kernel_size': {
            'values': [3, 5, 7]
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

sweep_id = wandb.sweep(svm_mlp_patches, project=wandb_config["project"], entity=wandb_config["entity"])
print(f"Initiated sweep with ID: {sweep_id}")
wandb.agent(sweep_id, function=run_experiment)
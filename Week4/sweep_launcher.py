import wandb
from run_experiment.run_experiment_cross_val import run_experiment
#from run_experiment.run_experiment_test import run_experiment

DO_HYPERPARAM_SEARCH = False
NUM_RUNS = 50

# Wandb configuration
wandb_config = {
    "project": "C3-Week4-mcvNet",
    "entity": "marc-org",
}

# Layout for perfoming the sweeps for the different experiments
baseline = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['baseline']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[False]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [4]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [4]
        },
        'save_weights': {
            'values': [False]
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

maxpool = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[False]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [4]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [4]
        },
        'save_weights': {
            'values': [False]
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

maxpool_data_aug = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[True]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [4]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [4]
        },
        'save_weights': {
            'values': [False]
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

# From here ALWAYS DATA AUG

maxpool_bn = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_bn']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[True]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [4]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [1]
        },
        'save_weights': {
            'values': [False]
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

maxpool_gap = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[True]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [4]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [4]
        },
        'save_weights': {
            'values': [False]
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

maxpool_gap_r = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_r']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[True]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [4]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [1]
        },
        'save_weights': {
            'values': [False]
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

attention = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_attention']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[True]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [2]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [1]
        },
        'save_weights': {
            'values': [False]
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

num_blocks = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap']
        },
        'optimizer': {
            'values':[("Adam")]
        },
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
        'data_aug': {
            'values':[True]
        },
        'init_chan': {
            'values': [20]
        },
        'num_blocks':{
            'values': [2]
        },
        'patience':{
            'values':[-1]
        },
        'min_delta':{
            'values':[0]
        },
        'k_folds': {
            'values': [4]
        },
        'save_weights': {
            'values': [False]
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


def run_experiment_with_wandb_config():
    """Wrapper function to pass wandb_config to run_experiment."""
    run_experiment(wandb_config=wandb_config)

if __name__ == "__main__":
    sweep_id = wandb.sweep(maxpool_bn, project=wandb_config["project"], entity=wandb_config["entity"])
    print(f"Initiated sweep with ID: {sweep_id}")

    if DO_HYPERPARAM_SEARCH:
        print("SEARCH MODE.")
        if NUM_RUNS is None:
            print(" Running forever (Infinite).")
            print("(Press Ctrl+C to stop manually)")
            wandb.agent(sweep_id, function=run_experiment)
        else:
            print(f"Running {NUM_RUNS} experiments.")
            wandb.agent(sweep_id, function=run_experiment, count=NUM_RUNS)
    else:
        print("NORMAL MODE.")
        wandb.agent(sweep_id, function=run_experiment)
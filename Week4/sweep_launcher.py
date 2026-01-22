import wandb
from run_experiment.run_experiment_cross_val import run_experiment
#from run_experiment.run_experiment_test import run_experiment

DO_HYPERPARAM_SEARCH = True
NUM_RUNS = None

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
            'values': [16]
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
            'values': [16]
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
            'values': [16]
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

maxpool_bn_data_aug = {
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
            'values': [16]
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

classifier_experiment = {
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
            'values': [16]
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
        'units_fc': {
            'values': [16, 32, 64, 128, 256]
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

maxpool_gap_bn = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_bn']
        },
        'optimizer': {
            'values':[("Adam")]
        },
        'image_size': {
            'values': [(224,224)]
        },
        'batch_size': {
            'values': [16]
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

maxpool_gap_bn_dw = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_bn_dw']
        },
        'optimizer': {
            'values':[("Adam")]
        },
        'image_size': {
            'values': [(224,224)]
        },
        'batch_size': {
            'values': [16]
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
            'values': [2,3,4,5]
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

number_of_filters = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_bn_dw']
        },
        'optimizer': {
            'values':[("Adam")]
        },
        'image_size': {
            'values': [(224,224)]
        },
        'batch_size': {
            'values': [16]
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
            'values': [5,10,12,15,20]
        },
        'num_blocks':{
            'values': [5]
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

maxpool_gap_bn_dw_at = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_bn_dw_at']
        },
        'optimizer': {
            'values':[("Adam")]
        },
        'image_size': {
            'values': [(224,224)]
        },
        'batch_size': {
            'values': [16]
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
            'values': [15]
        },
        'num_blocks':{
            'values': [5]
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

maxpool_gap_bn_dw_r = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_bn_dw_r']
        },
        'optimizer': {
            'values':[("Adam")]
        },
        'image_size': {
            'values': [(224,224)]
        },
        'batch_size': {
            'values': [16]
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
            'values': [15]
        },
        'num_blocks':{
            'values': [5]
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

image_size = {
    'method': 'grid',
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_bn_dw_p']
        },
        'optimizer': {
            'values':[("Adam")]
        },
        'image_size': {
            'values': [(224,224)]
        },
        'batch_size': {
            'values': [16]
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
            'values': [15]
        },
        'filters': {
            'values': [[]]
        },
        'num_blocks':{
            'values': [5]
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

hyperparam_search_random = {
    'method': 'random',
    'metric': {
        'name': 'distance',
        'goal': 'minimize'   
    },
    'parameters': {
        'block_type': {
            'values':['maxpool_gap_bn_dw_p']
        },
        'image_size': {
            'values': [(224,224)]
        },
        'output_dim': {
            'values': [8]
        },
        'data_aug': {
            'values':[True]
        },
        'init_chan': {
            'values': [15]
        },
        'filters': {
            'values': [[]]
        },
        'num_blocks':{
            'values': [5]
        },
        'patience':{
            'values':[10]
        },
        'min_delta':{
            'values':[0.001]
        },
        'k_folds': {
            'values': [4]
        },
        'save_weights': {
            'values': [False]
        },
        'num_workers': {
            'values': [8]
        },

        # --- HYPERPARAMETERS TO SEARCH ---
        
        # 1. Training Dynamics
        'batch_size': {
            'values': [8, 16, 32, 64]
        },
        'num_epochs': {
            'values': [20, 50, 100]
        },
        'learning_rate': {
            'values': [0.0001, 0.001, 0.01, 0.1]
        },
        
        # 2. Optimizer & Momentum
        'optimizer': {
            'values': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        },
        'momentum': {
            'values': [0.0, 0.5, 0.9]
        },

        # 3. Regularization & Topology
        'dropout_prob_1': { # Dropout convolutional layers
            'values': [0.0, 0.1, 0.2, 0.3] 
        },
        'dropout_prob_2': { # Dropout classifier layers
            'values': [0.0, 0.2, 0.3, 0.5] 
        },
        'weight_decay': { # Regularizers (L2 penalty)
            'values': [0.0, 0.000001, 0.00001, 0.0001, 0.001] 
        }
    }
}

def run_experiment_with_wandb_config():
    """Wrapper function to pass wandb_config to run_experiment."""
    run_experiment(wandb_config=wandb_config)

if __name__ == "__main__":
    sweep_id = wandb.sweep(hyperparam_search_random, project=wandb_config["project"], entity=wandb_config["entity"])
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
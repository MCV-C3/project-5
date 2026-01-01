import wandb
from run_experiment.run_experiment_cross_val import run_experiment
#from run_experiment.run_experiment_test import run_experiment

DO_HYPERPARAM_SEARCH = True
NUM_RUNS = 2

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
            'values':[0.001]
        },
        'save_weights':{
            'values':[False]
        },
        'k_folds':{
            'values': [4]
        },
        'add_aug':{
            'values': [True]
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
            'values': [list(range(15))]
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
        'add_aug':{
            'values': [True]
        }
    },
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'   
    }
}

hyperparam_search = {
    'method': 'bayes', 
    'metric': {
        'name': 'fold_accuracy_mean',
        'goal': 'maximize'
    },
    'parameters': {
        'blocks_to_keep': {
            'value': list(range(15)) # Blocks 0 to 14
        },
        'out_feat': {
            'value': -1
        },
        'feature_extraction': {
            'value': True
        },
        'image_size': {
            'value': (224, 224)
        },
        'k_folds': {
            'value': 4
        },

        # --- HYPERPARAMETERS TO SEARCH ---
        
        # 1. Training Dynamics
        'batch_size': {
            'values': [16, 32, 64, 128, 256, 512]
        },
        'num_epochs': {
            'values': [10, 50, 100]
        },
        'learning_rate': {
            'values': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        },
        
        # 2. Optimizer & Momentum
        'optimizer': {
            'values': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        },
        'momentum': {
            'values': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        },

        # 3. Regularization & Topology
        'dropout_prob': { # Drop-out layers params
            'values': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8] 
        },
        'weight_decay': { # Regularizers (L2 penalty)
            'values': [0.0, 0.0001, 0.001, 0.01] 
        },

        # 4. Data Augmentation
        'add_aug': {'value': True}, 

        # Other fixed parameters
        'num_workers': {'value': 8},
        'patience': {'value': 5},
        'min_delta': {'value': 0.001},
        'output_dim': {'value': 8},
        'save_weights': {'value': False}
    }
}

def run_experiment_with_wandb_config():
    """Wrapper function to pass wandb_config to run_experiment."""
    run_experiment(wandb_config=wandb_config)

if __name__ == "__main__":
    sweep_id = wandb.sweep(hyperparam_search, project=wandb_config["project"], entity=wandb_config["entity"])
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
import wandb
from run_experiment.run_experiment_cross_val import run_experiment


DO_NAS = True
NUM_RUNS = 40

wandb_config = {
    "project": "C3-Week4-NAS",
    "entity": "marc-org",
}


neural_architecture_search = {
    'method': 'bayes',
    'metric': {
        'name': 'distance',
        'goal': 'minimize' 
    },
    'parameters': {
        # ====== FIXED PARAMETERS ======
        'image_size': {
            'values': [(224, 224)]
        },
        'output_dim': {
            'values': [8]
        },
        'data_aug': {
            'values': [True]
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
        'patience': {
            'values': [10]
        },
        'min_delta': {
            'values': [0.001]
        },
        'filters': {
            'values': [[]]
        },

        
        # ====== ARCHITECTURE SEARCH SPACE ======
        
        # 1. Network Depth (number of blocks)
        'num_blocks': {
            'values': [3, 4, 5, 6, 7]
        },
        
        # 2. Network Width (initial channels)
        'init_chan': {
            'values': [12, 15, 18, 20, 24, 28, 32]
        },
        
        # 3. Architecture Components (different block types)
        'block_type': {
            'values': [
                'maxpool_gap_bn',           # Baseline with BN and pooling
                'maxpool_gap_bn_p',         # Baseline with BN and pooling + PReLU activation
                'maxpool_gap_bn_dw',        # Baseline with BN and pooling + Depthwise convolutions
                'maxpool_gap_bn_dw_p',      # Baseline with BN and pooling + Depthwise convolutions+ PReLU activation
            ]
        },
        

        
        # ====== BEST HYPERPARAMETERS FROM BAYES ======
        'batch_size': {
            'values': [64]
        },
        'num_epochs': {
            'values': [50]
        },
        'learning_rate': {
            'values': [0.0007955281528509873]
        },
        'optimizer': {
            'values': ['RMSprop']
        },
        'momentum': {
            'values': [0.7582520845191429]
        },
        'dropout_prob_1': {
            'values': [0.0024918879744524336]
        },
        'dropout_prob_2': {
            'values': [0.3886727784732117]
        },
        'weight_decay': {
            'values': [1.199948210802673e-06]
        }
    }
}


def run_experiment_with_wandb_config():
    """Wrapper function to pass wandb_config to run_experiment."""
    run_experiment(wandb_config=wandb_config)


if __name__ == "__main__":
    nas_config = neural_architecture_search
    
    sweep_id = wandb.sweep(nas_config, project=wandb_config["project"], entity=wandb_config["entity"])
    print(f"Neural Architecture Search initiated with ID: {sweep_id}")
    print(f"Project: {wandb_config['project']}")
    print(f"Method: {nas_config['method']}")

    if DO_NAS:
        print("NEURAL ARCHITECTURE SEARCH")
        if NUM_RUNS is None:
            print("   Running forever (Infinite).")
            print("   (Press Ctrl+C to stop manually)")
            wandb.agent(sweep_id, function=run_experiment)
        else:
            print(f"   Running {NUM_RUNS} architecture experiments.")
            wandb.agent(sweep_id, function=run_experiment, count=NUM_RUNS)
    else:
        print("SINGLE RUN MODE")
        wandb.agent(sweep_id, function=run_experiment, count=1)

import wandb
from run_experiment.run_experiment_cross_val import run_experiment as run_experiment_cross_val
from run_experiment.run_experiment_test import run_experiment as run_experiment_test

if __name__ == "__main__":
    default_experiment_config = {
        "image_size": (224, 224),
        "batch_size": 256,
        "num_epochs": 20,
        "learning_rate": 0.001,
        "output_dim": 8,
        "num_workers": 8,
        "patience": 5,
        "min_delta": 0,
        "save_weights": True,
        "k_folds": 4
    }
    experiments = [
        # {"experiment_name": "baseline"}, # All augs at 0.0
        # {"experiment_name": "aug_flip", "aug_horizontal_flip": 0.5},
        {"experiment_name": "aug_rotation", "aug_rotation": 0.5},
        {"experiment_name": "aug_jitter", "aug_color_jitter": 0.5},
        {"experiment_name": "aug_zoom", "aug_zoom": 0.5},
        {"experiment_name": "aug_blur", "aug_gaussian_blur": 0.5},
        # {"experiment_name": "imagenet_norm", "use_imagenet_norm":True}
    ]
    experiments = [
        {"experiment_name": "baseline_test", 
         "aug_rotation": 0.5, 
         "aug_zoom": 0.5, 
         "aug_horizontal_flip": 0.5, 
         "aug_color_jitter": 0.5,
         "aug_zoom": 0.5,
         "aug_gaussian_blur": 0.5,
         "use_imagenet_norm":True,
         "add_aug":True
         },
    ]

    experiments = [
    # {
    #     "experiment_name": "baseline", 
    #     "add_aug": False, 
    #     "use_imagenet_norm": True
    # },
    {
        "experiment_name": "expanded_flip", 
        "aug_horizontal_flip": 0.5, 
        "add_aug": True, 
        "use_imagenet_norm": True
    },
    {
        "experiment_name": "expanded_rotation", 
        "aug_rotation": 0.3, 
        "add_aug": True, 
        "use_imagenet_norm": True
    },
    {
        "experiment_name": "expanded_jitter", 
        "aug_color_jitter": 0.2, 
        "add_aug": True, 
        "use_imagenet_norm": True
    },
    {
        "experiment_name": "expanded_zoom", 
        "aug_zoom": 0.2, 
        "add_aug": True, 
        "use_imagenet_norm": True
    },
    {
        "experiment_name": "expanded_blur", 
        "aug_gaussian_blur": 0.2, 
        "add_aug": True, 
        "use_imagenet_norm": True
    }
]

    experiments = [
        # {
        #     "experiment_name": "baseline", 
        #     "add_aug": False, 
        #     "use_imagenet_norm": True
        # },
        {
            "experiment_name": "expanded_flip", 
            "aug_horizontal_flip": 0.5, 
            "add_aug": True, 
            "use_imagenet_norm": False
        },
        {
            "experiment_name": "expanded_rotation", 
            "aug_rotation": 0.3, 
            "add_aug": True, 
            "use_imagenet_norm": False
        },
        {
            "experiment_name": "expanded_jitter", 
            "aug_color_jitter": 0.2, 
            "add_aug": True, 
            "use_imagenet_norm": False
        },
        {
            "experiment_name": "expanded_zoom", 
            "aug_zoom": 0.2, 
            "add_aug": True, 
            "use_imagenet_norm": False
        },
        {
            "experiment_name": "expanded_blur", 
            "aug_gaussian_blur": 0.2, 
            "add_aug": True, 
            "use_imagenet_norm": False
        }
    ]

    experiments = [
        {
            "experiment_name": "expanded_all_augs_imagenet_norm",
            "aug_horizontal_flip": True,
            "aug_rotation": True,
            "aug_color_jitter": True,
            "aug_zoom": True,
            "aug_gaussian_blur": True,
            "add_aug": True,
            "use_imagenet_norm": False
        }
    ]
    experiments = [
        {
        
        "experiment_name": "swin_finetune",
        "backbone": "swin",
        "feature_extraction": False,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "use_imagenet_norm": True,
        }
        # "dropout_prob": 0.1

    ]    
    
    experiments_small = [ # best params small dataset (kfold) bayes search
        {
        "batch_size": 64,
        "dropout_prob": 0.11002281839218377,
        "learning_rate": 7.00834290308464e-05,
        "momentum": 0.1523167953194062,
        "num_epochs": 50,
        "optimizer": "Nadam",
        "weight_decay": 2.7431020646302626e-05,
        "feature_extraction": False,
        "add_aug": True,
        "aug_horizontal_flip": True,
        "aug_rotation": True,
        "aug_color_jitter": True,
        "aug_zoom": True,
        "aug_gaussian_blur": True,
        "use_imagenet_norm": True,
        "backbone": "mobilenet",
        
        "experiment_name": "best_model_expanded_all_augs_imagenet_norm",
        }
    ]
    
    experiments = [ # best params big dataset (test)
        {
        "batch_size": 256,
        "dropout_prob": 0.0,
        "learning_rate": 0.001,
        "momentum": 0.0,
        "num_epochs": 20,
        "optimizer": "Adam",
        "weight_decay": 0,
        "feature_extraction": False,
        "patience": 20,
        "use_imagenet_norm": True,
        "backbone": "mobilenet",
        
        "experiment_name": "best params big dataset (test)",
        }
    ]

    EXPERIMENT_TYPE = "test"  # "cross_val" or "test"

    for exp_cfg in experiments_small:
        config = {**default_experiment_config, **exp_cfg}
        print(f"\n🚀 LAUNCHING EXPERIMENT: {config['experiment_name']}")
        if EXPERIMENT_TYPE == "cross_val":
            run_experiment_cross_val(experiment_config=config)
        elif EXPERIMENT_TYPE == "test":
            run_experiment_test(experiment_config=config)

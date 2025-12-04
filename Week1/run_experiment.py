import wandb

from main import train, test, Dataset, SPLIT_PATH
from bovw import BOVW

def run_experiment():
    
    config_defaults = {
        "detector_type": "SIFT",
        "codebook_size": 50,
        "dataset_path": SPLIT_PATH,
    }

    # Init wandb
    wandb.init(
        project="C3-Week1-BOVW",
        entity="project-5",
        config=config_defaults,
    )

    cfg = wandb.config

    print(f"Starting experiment with config: {cfg}")
    
    print("Loading datasets...")
    data_train = Dataset(ImageFolder=SPLIT_PATH + "train")
    data_test = Dataset(ImageFolder=SPLIT_PATH + "test")

    det_kwargs = {}
    if 'nfeatures' in cfg:
        det_kwargs['nfeatures'] = cfg.nfeatures

    cb_kwargs = {}
    
    bovw = BOVW(
        detector_type=cfg.detector_type,
        codebook_size=cfg.codebook_size,
        detector_kwargs=det_kwargs,
        codebook_kwargs=cb_kwargs
    )

    print("Training the model...")
    bovw, classifier, train_acc = train(dataset=data_train, bovw=bovw)

    print("Evaluating the model...")
    test_acc = test(dataset=data_test, bovw=bovw, classifier=classifier)

    wandb.log({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "detector_type": cfg.detector_type,
        "n_words": cfg.codebook_size,
    })

    print(f"Experiment completed. Train Acc: {train_acc}, Test Acc: {test_acc}")

    wandb.finish()

if __name__ == "__main__":
    run_experiment()




import logging
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from .training import ClassifierTrainer
from classifier import VanillaClassifier
from .utils import set_random_seeds, get_fmnist_functa
import itertools
import random

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f"hyperparameter_tuning.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")
    #assert(device == 'cuda')
    # non config params
    data_path = '/datasets/functaset'
    num_classes = 10
    in_features = 512
    checkpoint_file = 'checkpoints/mlp'
    num_epochs = 500
    loss_fn = nn.CrossEntropyLoss()


    param_grid = {
        'lr': [1e-4, 1e-3],
        'hidden_sizes': [
            [512, 256, 128],
            [256, 128],
            [128, 128],
            [512,512,512],
            [128,64]

        ],
        'p_dropout': [None, 0.1, 0.3],
        'batch_size': [32,64,128,256],
        'normalization': ["", "layer", "batch"],
        'activation': ["relu", "lrelu"],
        'early_stopping': [5, 10]
    }

    # To store the best results
    best_val_accuracy = 0.0
    best_params = None

    # Generate all combinations of hyperparameters
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Random search: Define number of random combinations to test
    random_sample_size = 150 # For example, test 100 randomly selected combinations

    # Randomly sample parameter combinations
    random_combinations = random.sample(param_combinations, random_sample_size)

    # Iterate over all hyperparameter combinations
    for idx, params in enumerate(random_combinations):
        try:
            logger.info(f"Testing combination {idx + 1}/{len(random_combinations)}: {params}")

            set_random_seeds(0)

            train_functaloader = get_fmnist_functa(data_dir=f"{data_path}/fmnist_train.pkl", mode='train',
                                                   batch_size=params['batch_size'], num_workers=2)
            val_functaloader = get_fmnist_functa(data_dir=f"{data_path}/fmnist_val.pkl", mode='test',
                                                 batch_size=params['batch_size'], num_workers=2)

            hidden_dims = params['hidden_sizes']
            activations = [params['activation']] * len(hidden_dims)
            model = VanillaClassifier(in_features=in_features, num_classes=num_classes, hidden_dims=hidden_dims,
                                      nonlins=activations, p_dropout=params['p_dropout'],
                                      normalization=params['normalization']).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.1, patience=3, verbose=True
            )

            def post_epoch_fn(epoch, train_res, test_res, verbose):
                scheduler.step(test_res.accuracy)

            trainer = ClassifierTrainer(model, loss_fn, optimizer, device)
            fit_res = trainer.fit(train_functaloader, val_functaloader, num_epochs, max_batches=None,
                                  post_epoch_fn=post_epoch_fn, early_stopping=params['early_stopping'],
                                  checkpoints=checkpoint_file + f"{params}", print_every=None)

            curr_best_val_acc = fit_res.test_acc[fit_res.last_checkpoint_idx]
            logger.info(f"curr best accuracy: {curr_best_val_acc}, params: {params}")
            if best_val_accuracy < curr_best_val_acc:
                best_val_accuracy = curr_best_val_acc
                best_params = params
                logger.info(f"New best accuracy: {best_val_accuracy}, params: {params}")

        except Exception as e:
            logger.error(f"Error with params {params}: {e}")
        # Log the final best parameters
    logger.info(f"Best Validation Accuracy: {best_val_accuracy}")
    logger.info(f"Best Hyperparameters: {best_params}")

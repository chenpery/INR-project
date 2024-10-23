import os
import sys
import pathlib
import urllib
import shutil
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from classifier import evaluate_model, VanillaClassifier
from src.utils import set_random_seeds, vec_to_img, get_fmnist_functa
from src.training import  AdversarialTrainer
import argparse
from attack import attack_classifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a weight-space classifier')
    parser.add_argument('-p', '--data-path', type=str, default='/datasets/functaset',
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-c', '--cpu', action='store_true', help="If set, use cpu and not cuda")
    # add any other parameters you may need here
    args = parser.parse_args()

    # Set random seed.
    set_random_seeds(0)
    device = 'cpu' if args.cpu else 'cuda:0'

    # Handle data-loading - these loaders yield(vector,label) pairs.
    train_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_train.pkl", mode='train',
                                           batch_size=args.batch_size, num_workers=2)
    val_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_val.pkl", mode='test',
                                         batch_size=args.batch_size, num_workers=2)
    test_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_test.pkl", mode='test',
                                          batch_size=args.batch_size, num_workers=2)


    num_classes = 10
    in_features = 512
    checkpoint_file = 'checkpoints/robust_mlp'
    num_epochs = 500
    early_stopping = 10
    lr = 1e-3
    hidden_dims = [512, 256, 128]
    p_dropout = 0.1
    normalization = "batch"
    activations = ["lrelu"] * len(hidden_dims)
    loss_fn = nn.CrossEntropyLoss()

    # Instantiate Classifier Model
    model = VanillaClassifier(in_features=in_features, num_classes=num_classes, hidden_dims=hidden_dims,
                              nonlins=activations, p_dropout=p_dropout,
                              normalization=normalization).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)


    def post_epoch_fn(epoch, train_res, test_res, verbose):
        scheduler.step(test_res.accuracy)


    trainer = AdversarialTrainer(model, loss_fn, optimizer, device=device, randomize=False, linf_bound=0.01,
                                 num_pgd_steps=10)

    # Train, unless final checkpoint is found
    checkpoint_file_final_prefix = f'{checkpoint_file}_final'
    checkpoint_file_final = f'{checkpoint_file_final_prefix}.pt'
    if os.path.isfile(checkpoint_file_final):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        saved_state = torch.load(checkpoint_file_final, map_location=device)
        model.load_state_dict(saved_state['model_state'])
    else:
        try:

            fit_res = trainer.fit(train_functaloader, val_functaloader, num_epochs, max_batches=None,
                                  post_epoch_fn=post_epoch_fn, early_stopping=early_stopping,
                                  checkpoints=checkpoint_file_final_prefix, print_every=None)

            # load best model
            if os.path.isfile(checkpoint_file_final):
                saved_state = torch.load(checkpoint_file_final, map_location=device)
                model.load_state_dict(saved_state['model_state'])

        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')

    train_accuracy = evaluate_model(model, train_functaloader, device)
    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')

    validation_accuracy = evaluate_model(model, val_functaloader, device)
    print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')

    test_accuracy = evaluate_model(model, test_functaloader, device)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


    test_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_test.pkl", mode='test',
                                          batch_size=1, num_workers=2)

    linf_bounds = [10 ** (-i) for i in range(3, 7)] + [5 * 10 ** (-i) for i in range(3, 7)]
    linf_bounds.sort()
    accuracies = []
    for bound in linf_bounds:
        print(f'Attacking with linf_bound={bound}')
        all_labels, all_preds = attack_classifier(model, test_functaloader, loss_fn, bound)
        accuracy = accuracy_score(all_labels, all_preds) * 100
        accuracies.append(accuracy)
        print(f'Accuracy after attack with linf_bound={bound}: {accuracy:.2f}%')
    print(accuracies)
"""
In this script you should train your 'clean' weight-space classifier.
"""

import os
import torch
import torch.nn as nn
from utils import set_random_seeds, vec_to_img, get_fmnist_functa
import numpy as np
from SIREN import ModulatedSIREN
import argparse
from mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS
import torch.optim as optim
from training import ClassifierTrainer


# A basic linear classifier.
class VanillaClassifier(nn.Module):
    def __init__(self, in_features=512, num_classes=10, hidden_dims=None, nonlins=None, p_dropout=None, normalization=""):
        """
        :param in_features: input_dimension.
        :param num_classes: number of classes (output dimension).
        """
        super(VanillaClassifier, self).__init__()
        if hidden_dims is None:
            hidden_dims=[]
        if nonlins is None:
            nonlins=[]
        self.net= MLP(in_dim=in_features, dims=hidden_dims+[num_classes], nonlins=nonlins +["none"],
                      p_dropout=p_dropout, normalization=normalization)
    def forward(self, x):
        return self.net(x)


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Disable gradient calculation during inference
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Get predicted class (argmax over the outputs)
            predicted = torch.argmax(y_pred, dim=1)

            # Accumulate the total and correct predictions
            correct += (predicted == y).sum().item()
            total += y.size(0)

    # Calculate accuracy
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a weight-space classifier')
    parser.add_argument('-p', '--data-path', type=str, default='/datasets/functaset',
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-c', '--cpu', action='store_true', help = "If set, use cpu and not cuda")
    # add any other parameters you may need here
    args = parser.parse_args()

    # Set random seed.
    set_random_seeds(0)
    device = 'cpu' if args.cpu else 'cuda:0'

    # Handle data-loading - these loaders yield(vector,label) pairs.
    train_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_train.pkl",mode='train', batch_size = args.batch_size, num_workers = 2) 
    val_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_val.pkl",mode='test', batch_size = args.batch_size, num_workers = 2)
    test_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_test.pkl",mode='test', batch_size = args.batch_size, num_workers = 2)
    
    # Load Full INR - this is only for visualization purposes - this is just an example, you can erase this when you submit
    #inr = ModulatedSIREN(height=28, width=28, hidden_features=256, num_layers=10, modul_features=512)
    #inr.load_state_dict(torch.load(f"{args.data_path}/modSiren.pth")['state_dict'])
    #inr = inr.to(device)
    
    #Example of extracting full image from modulation vector - must pass a single (non-batched) vector input - this is just an example, you can erase this when you submit
    #img = vec_to_img(inr, train_functaloader.dataset[0][0].to(device))

    # TODO: Implement your training and evaluation loops here. We recommend you also save classifier weights for next parts
    num_classes = 10
    in_features = 512
    checkpoint_file = 'checkpoints/mlp'
    num_epochs = 500
    early_stopping = 5
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

    trainer = ClassifierTrainer(model, loss_fn, optimizer, device)

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

            #load best model
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


    #inference example
    #predicted_scores = classifier(train_functaloader.dataset[0][0].to(device))
    

    
  
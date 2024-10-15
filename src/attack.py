from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import set_random_seeds, vec_to_img, get_fmnist_functa
import numpy as np
from SIREN import ModulatedSIREN
from classifier import VanillaClassifier
import argparse
from sklearn.metrics import accuracy_score


def attack_classifier(model,
                      loader: DataLoader,
                      criterion,
                      linf_bound,
                      num_pgd_steps=10,
                      device="cuda"):
    """
    :param model: your trained classifier model
    :param loader: data loader for input to be perturbed
    :param criterion: The loss criteria you wish to maximize in attack

    1. Initialization: setting up perturbations and the optimizer.
    2. Optimization loop (PGD Steps): Iteratively adjusting perturbations to maximize loss.
    3. Projection: ensuring perturbations remain within specified bounds.
    4. Evaluation: assessing the model's accuracy on perturbed inputs.
    """

    model.eval()  # Model should be used in evaluation mode - we are not training any model weights.
    all_preds = []
    all_labels = []
    prog_bar = tqdm(loader, total=len(loader))
    for vectors, labels in prog_bar:
        vectors, labels = vectors.to(device), labels.to(device)
        perts = torch.zeros_like(vectors)  # initialize the perturbation vectors for current iteration

        ''' TODO (1): Your perts tensor currently will not be optimized since torch wasn't instructed to track gradients for it - make torch track its gradients. '''
        perts.requires_grad = True

        ''' TODO (2): Initialize your optimizer, you might need to finetune the learn-rate.
        What should be the set of parameters the optimizer will be changing? Hint: NOT model.parameters()!
        '''
        attack_learning_rate = linf_bound / 4
        # optimizer = optim.Adam([perts], lr=attack_learning_rate)
        optimizer = optim.RMSprop([perts], lr=attack_learning_rate, alpha=0.99)

        '''Every step here is one PGD iteration (meaning, one attack optimization step) optimizing your perturbations.
        After the loop below is over you'd have all fully-optimized perturbations for the current batch of vectors.'''
        for step in range(num_pgd_steps):
            preds = model(vectors + perts)  # feed currently perturbed data into the model
            ''' TODO (3):  What's written in this line for the loss is almost correct. Change the code to MAXIMIZE the loss'''
            # optimizer minimize the loss and we want to maximize the loos to make the model misclassify
            # minimizing -loss is equivalent to maximizing the loss
            loss = -criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ''' TODO (4): Perform needed L_inf norm bound projection. The 'torch.clamp' function could be useful.'''
            # applying the L∞ norm constraint
            # ensures that each element of perts remains within the range [-linf_bound, linf_bound]
            # constraint should not influence gradient calculations.
            with torch.no_grad():
                epsilon = 1e-10
                perts.clamp_(-linf_bound + epsilon, linf_bound - epsilon)

            assert perts.abs().max().item() <= linf_bound  # If this assert fails, you have a mistake in TODO(4)
            perts = perts.detach().requires_grad_()  # Reset gradient tracking - we don't want to track gradients for norm projection.

        ''' TODO (5): Accumulate predictions and labels to compute final accuracy for the attacked classifier.
        You can compute final predictions by taking the argmax over the softmax of predictions.'''
        with torch.no_grad():
            perturbed_vectors = vectors + perts
            outputs = model(perturbed_vectors)
            # outputs has a shape of (batch_size, num_classes)
            # for determine the predicted class
            _, predicted = torch.max(outputs, dim=1)

        # for each batch.
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize an adversarial attack over a pre-trained weight-space classifier')
    parser.add_argument('-p', '--data-path', type=str, default='/datasets/functaset',
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-c', '--cpu', action='store_true', help = "If set, use cpu and not cuda")
    parser.add_argument('-m', '--model-path', type=str, default='checkpoints/mlp_final.pt', help="Path to your pretrained classifier model weights")
    # add any other parameters you may need here
    args = parser.parse_args()
    
    # Set random seed.
    set_random_seeds(0)
    device = 'cpu' if args.cpu else 'cuda:0'

    # the same parameters as used during training
    hidden_dims = [512, 256, 128]
    activations = ["lrelu"] * len(hidden_dims)
    p_dropout = 0.1
    normalization = "batch"


    # Instantiate Classifier Model and load weights
    classifier = VanillaClassifier(
        in_features=512,
        num_classes=10,
        hidden_dims=hidden_dims,
        nonlins=activations,
        p_dropout=p_dropout,
        normalization=normalization
    ).to(device)
    saved_state = torch.load(args.model_path, map_location=device)
    classifier.load_state_dict(saved_state['model_state'])

    test_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_test.pkl", mode='test',
                                          batch_size=args.batch_size, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    linf_bounds = [10**(-i) for i in range(3,7)] + [5*10**(-i) for i in range(3,7)]
    linf_bounds.sort()
    accuracies = []
    for bound in linf_bounds:
        print(f'Attacking with linf_bound={bound}')
        all_labels, all_preds = attack_classifier(classifier, test_functaloader, criterion, bound)
        accuracy = accuracy_score(all_labels, all_preds) * 100
        accuracies.append(accuracy)
        print(f'Accuracy after attack with linf_bound={bound}: {accuracy:.2f}%')
    print(accuracies)
import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader
from torch import optim
from .train_results import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(
            self,
            dl_train: DataLoader,
            dl_test: DataLoader,
            num_epochs,
            checkpoints: str = None,
            early_stopping: int = None,
            print_every=1,
            post_epoch_fn=None,
            **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        checkpoint_idx = None
        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get("best_acc", best_acc)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if print_every is not None and (epoch % print_every == 0 or epoch == num_epochs - 1):
                verbose = True
            self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

            # TODO:
            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            # train epoch
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            train_loss.append(sum(train_result.losses) / len(train_result.losses))
            train_acc.append(train_result.accuracy)
            # train test
            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss.append(sum(test_result.losses) / len(test_result.losses))
            test_acc.append(test_result.accuracy)

            actual_num_epochs = actual_num_epochs + 1

            # early stopping
            if best_acc is None or test_result.accuracy > best_acc:
                # ====== YOUR CODE: ======
                epochs_without_improvement = 0
                best_acc = test_result.accuracy
                save_checkpoint = True
                # ========================
            else:
                # ====== YOUR CODE: ======
                epochs_without_improvement = epochs_without_improvement + 1
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    break
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                checkpoint_idx= epoch
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch + 1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc, checkpoint_idx)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            verbose=True,
            max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.2f})"
            )

        return EpochResult(losses=losses, accuracy=accuracy)


class ClassifierTrainer(Trainer):

    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        y_pred = self.model.forward(X)
        loss = self.loss_fn(y_pred, y)
        # Backward pass

        loss.backward()
        # Update parameters
        self.optimizer.step()

        # calculate number of correct predictions
        predicted_classes = torch.argmax(y_pred, dim=1)
        num_correct = (predicted_classes == y).sum().item()
        batch_loss = float(loss)
        # ========================

        return BatchResult(batch_loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        batch_loss: float
        num_correct: int

        with torch.no_grad():
            y_pred = self.model.forward(X)
            loss = self.loss_fn(y_pred, y)
            predicted_classes = torch.argmax(y_pred, dim=1)
            num_correct = (predicted_classes == y).sum().item()
            batch_loss = float(loss)
            # ========================

        return BatchResult(batch_loss, num_correct)



class AdversarialTrainer(ClassifierTrainer):
    def __init__(self, model, loss_fn, optimizer, linf_bound=1e-6, num_pgd_steps=10, randomize=False, device=None):
        super().__init__(model, loss_fn, optimizer, device)
        self.linf_bound = linf_bound
        self.num_pgd_steps = num_pgd_steps
        self.randomize = randomize

    def attack(self, X, y):
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        if self.randomize:
            perts = torch.rand_like(X, requires_grad=True)
            perts.data = perts.data * 2 * self.linf_bound - self.linf_bound
        else:
            perts = torch.zeros_like(X, requires_grad=True)

        attack_learning_rate = self.linf_bound / 4

        # Perform PGD steps
        for step in range(self.num_pgd_steps):
            logits = self.model(X + perts)
            loss = self.loss_fn(logits, y)
            loss.backward()
            perts.data = (perts + attack_learning_rate*perts.grad.detach()).clamp(-self.linf_bound, self.linf_bound)
            perts = perts.detach().requires_grad_()

        return perts.detach()  # Return adversarial examples

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # Generate adversarial examples
        self.model.eval()  # Keep BN layers in eval mode for adversarial attack generation
        pert = self.attack(X, y)
        self.model.train()  # Return to train mode

        # Concatenate clean and adversarial examples
        X_combined = torch.cat([X, X + pert], dim=0)
        y_combined = torch.cat([y, y], dim=0)

        # Forward pass on concatenated batch
        combined_y_pred = self.model.forward(X_combined)

        # Split predictions for clean and adversarial examples
        clean_y_pred, adv_y_pred = torch.split(combined_y_pred, X.size(0), dim=0)

        # Calculate individual losses
        clean_loss = self.loss_fn(clean_y_pred, y)
        adv_loss = self.loss_fn(adv_y_pred, y)

        # Total loss with weighted contributions
        tot_loss = 0.5 * adv_loss + 0.5 * clean_loss

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        tot_loss.backward()
        self.optimizer.step()

        # Calculate the number of correct predictions for clean data
        predicted_classes = torch.argmax(clean_y_pred, dim=1)
        num_correct = (predicted_classes == y).sum().item()
        batch_loss = float(clean_loss)

        return BatchResult(batch_loss, num_correct)



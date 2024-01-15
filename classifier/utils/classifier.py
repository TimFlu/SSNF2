# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
# other
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import pickle as pkl
from utils.datasets import CustomDataset
# self written
from utils.plots_classifier import plot_loss_function, plot_data, roc_plot, feature_importance
from utils.models import SimpleNN
from utils.ROC_analysis_2 import ROC_analysis, correct_mc
# logging
from utils.log import setup_comet_logger
import logging
logger = logging.getLogger(__name__)

# **************** Early Stopping ***************** #
class EarlyStopping:
    """
    Early stopping during training to avoid overfitting

    Attributes:
    patience (int): how many times in row the early stopping condition was not fullfilled
    mind_delta (float): how big the relative loss between two consecutive trainings must be
    counter (int): how many consecutive times the stopper was triggered
    best_loss (float): the best loss achieved in the training so far, used to calculate the relative loss
    early_stop (bool): False if stopper did not reach the patience and the training should continue. True if training should end.

    Methods:
    __call__: Determines if the current loss is correcting the model enough or not and if the training should end or not.
    """
    def __init__(self, patience=5, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = 10e10
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Determines if the current loss is correcting the model enough or not and if the training should end or not.

        Input:
        val_loss: the current loss of the training
        """
        relative_loss = (self.best_loss - val_loss) / self.best_loss * 100
        logger.info(f"Early stopping relative loss = {relative_loss}")
        if relative_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif relative_loss < self.min_delta:
            self.counter += 1
            logger.info(
                f"Early stopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                logger.info("Early stopping")
                self.early_stop = True

# ********* Create training and test data ********* #
def create_data(device, cfg, logger):
    """
    Creates the data used to train the classifier on.

    Parameters:
    device: current device CUDA or CPU
    cfg: configs
    logger: comet_logger used to overview the training

    Returns:
    mc_and_data_dataloader_train (DataLoader): Mixed samples of MC and data used to train the classifier.
    mc_and_data_dataloader_test (DataLoader): Mixed samples of MC and data used to test classifier.
    mc_corr_and_data_dataloader_train (DataLoader): Mixed samples of MC corrected and data used to train classifier.
    mc_corr_and_data_dataloader_trest (DataLoader): Mixed samples of MC corrected and data used to test classifier.
    """

    if cfg.data.name == "sonar":
        sonar_data = pd.read_csv("/work/tfluehma/git/SSNF2/classifier/data/sonar_data.csv")
        X = sonar_data.iloc[:, 0:60]
        X["weight"] = [1 for i in range(len(X))]
        y = sonar_data.iloc[:, 60]

        # Change string labels to integers
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        # Convert the data into pytorch tensors
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
        # Create Dataset
        train_dataset_sonar = CustomDataset(X_train, y_train, device=device)
        test_dataset_sonar = CustomDataset(X_test, y_test, device=device)
        # Create DataLoader
        batch_size = cfg.hyperparameters.batch_size
        train_dataloader = DataLoader(train_dataset_sonar, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset_sonar, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader, train_dataloader, test_dataloader

    else:
    #     train_data = pd.read_parquet("/work/tfluehma/git/SSNF2/classifier/data/train_data.parquet")
    #     test_data = pd.read_parquet("/work/tfluehma/git/SSNF2/classifier/data/test_data.parquet")

    #     train = train_data.iloc[:, :-1]
    #     train_label = train_data["label"]

    #     test = test_data.iloc[:, :-1]
    #     test_label = test_data["label"]

    #     # Change string labels to integers
    #     encoder = LabelEncoder()
    #     encoder.fit(train_label)
    #     train_label = encoder.transform(train_label)
    #     test_label = encoder.transform(test_label)

    #     # Define the pipelines
    #     with open("/work/tfluehma/git/SSNF2/preprocess/pipelines_eb.pkl", "rb") as file:
    #         pipelines = pkl.load(file)
    #         pipelines = pipelines["pipe1"]

    #     # Create Dataset
    #     train_dataset = CustomDataset(train, train_label, pipelines=pipelines, target_only=cfg.data.target_only, device=device)
    #     test_dataset = CustomDataset(test, test_label, pipelines=pipelines, target_only=cfg.data.target_only, device=device)

    #     # Plot the data to verify preprocessing worked
    #     plot_data(train_dataset.data, comet_logger=logger, cfg=cfg, keys=train.keys(), name="")

    #     # Create Dataloader
    #     batch_size = cfg.hyperparameters.batch_size
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # correct MC TODO: Copy ROC_analysis_2 here.
        mc_and_data_dataloader_train, mc_corr_and_data_dataloader_train = correct_mc(cfg, logger, device, dataset="train")
        mc_and_data_dataloader_test, mc_corr_and_data_dataloader_test = correct_mc(cfg, logger, device, dataset="test")

    return mc_and_data_dataloader_train, mc_and_data_dataloader_test, mc_corr_and_data_dataloader_train, mc_corr_and_data_dataloader_test

# **************** Train Function ***************** #
def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Training step of the classifier.

    Parameters:
        dataloader: test dataset to train on.
        mode: current model
        loss_fn: the loss function
        optimizer: the optimizer
    
    Returns:
        avg_batch_loss: The averaged loss per batch.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode
    model.train()
    epoch_loss = 0.0
    for batch, (X, y, weight) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        loss = torch.mean(weight*loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch % 5000 == 0:
            testing_loss, current = loss.item(), (batch + 1) * len(X)
            correct = (pred.round() == y).float().mean()
            logger.info(f"current batch loss: {testing_loss:>7f} Accuracy: {(100*correct):>0.1f}%  [{current:>5d}/{size:>5d}]")
    avg_batch_loss = epoch_loss / num_batches
    return avg_batch_loss

# ***************** Test Function ***************** #
def test_loop(dataloader, model, loss_fn):
    """
    The testing step of classifier.

    Parameters:
        dataloader:
        model:
        loss_fn:
    
    Returns:
        avg_batch_test_loss (float): The averaged loss over the amount of batches.
        correct (float): Percentage of how many guesses the model predicted correctly.
    """
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_test_loss, correct = 0, 0

    # Evaluation the model with torch.no_grad() ensures that no gradients
    # are computed during test mode
    with torch.no_grad():
        for X, y, _ in dataloader:
            pred = model(X)
            epoch_test_loss += torch.mean(loss_fn(pred, y)).item()
            correct += (pred.round() == y).float().sum()

    avg_batch_test_loss = epoch_test_loss / num_batches
    correct /= size
    logger.info(f"Avg Test loss per batch: {avg_batch_test_loss:>8f}, Total Accuracy: {(100*correct):>0.1f}% \n")
    
    return avg_batch_test_loss, 100*correct

# ******** classifier function containing training, testing and keeping track of results ********
def classify(device, cfg, comet_logger=None):
    """
    Training of Binary Classifier on classifying Monte Carlo sampes and data samples

    Parameters:
    device: Cuda or CPU
    cfg: config according to test_config.yaml

    Returns:
    """
    
    # Setup Comet logger
    # if cfg.logger and comet_logger is not None:
    #     comet_name = os.getcwd().split("/")[-1]
    #     comet_logger = setup_comet_logger(comet_name, cfg)
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=cfg.stopper.patience, min_delta=cfg.stopper.min_delta)
    best_test_loss = 100000000
    # create the datasets
    train_uncorr_dataloader, test_uncorr_dataloader, train_corrected_dataloader, test_corrected_dataloader = create_data(device, cfg, comet_logger)
    if cfg.data.corrected:
        train_dataloader = train_corrected_dataloader
        test_dataloader =  test_corrected_dataloader
    else:
        train_dataloader = train_uncorr_dataloader
        test_dataloader = test_uncorr_dataloader

    # create model
    input_size = train_dataloader.dataset.data.shape[1]
    num_layers = cfg.model.num_layers
    hidden_size = cfg.model.hidden_size
    model = SimpleNN(input_size, hidden_size, num_layers).to(device)
    logger.info("Training with Model: \n{}".format(model))
    # Define Hyperparameters
    learning_rate = cfg.hyperparameters.learning_rate
    weight_decay = cfg.hyperparameters.weight_decay
    epochs = cfg.hyperparameters.epochs

    # initialize the loss function and optimizer
    loss_fn = nn.BCELoss(reduction='none')
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.error(f"Optimizer {cfg.optimzer} not defined here. Use Adam or SGD")

    # **** Train and Test ****
    # keep track of models loss
    train_loss_list = []
    test_loss_list = []
    test_accuracy_list = []
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        # loss is averaged per batch
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn)
        if cfg.logger:
            comet_logger.log_metrics({"avg_batch_train_loss": train_loss, "avg_batch_test_loss": test_loss,
                                       "test_accuracy": test_accuracy}, step=t)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        # save the best model
        if test_loss < best_test_loss:
            logger.info("New best test loss, saving model...")
            best_test_loss = test_loss
            torch.save(model.state_dict(), "./best_model_weights.pth")
        # save the latest model
        logger.info("Saving latest model...")
        torch.save(model.state_dict(), "./latest_model_weights.pth")
        # Check if stopping early
        early_stopping(train_loss)
        if early_stopping.early_stop:
            break
        # plot ROC curve and feature importance every 10 epchs
        if t % 10 == 0:
            roc_plot(test_dataloader, cfg, comet_logger, device)
            # plot feature importance
            feature_importance(model, test_dataloader.dataset.data, cfg, comet_logger, device)
            
    # plot the loss functions and final ROC curve
    plot_loss_function(training_loss=train_loss_list, testing_loss=test_loss_list, comet_logger=comet_logger, cfg=cfg)
    roc_plot(test_dataloader, cfg, comet_logger, device)
    
    if cfg.logger and cfg.data.name != "sonar":
        ROC_analysis(cfg, device, comet_logger)

    elif cfg.data.name == "sonar":
        model.load_state_dict(torch.load("./best_model_weights.pth"))
        model.eval()
        X_tensor = torch.tensor(test_dataloader.dataset.data, dtype=torch.float32, requires_grad=True, device=device)
        output = model(X_tensor)
        output.backward(torch.ones_like(output))
        feature_importance_= X_tensor.grad.abs().mean(dim=0).cpu()
        # Plotting feature importances
        fig = plt.figure(figsize=(12, 12))
        plt.bar(range(len(feature_importance_)), feature_importance_, align='center')
        plt.xlabel('Feature')
        plt.ylabel('Importance Score')
        plt.title('Feature Importances')
        plt.savefig("./plots/feature_importance.png")
        if cfg.logger:
            comet_logger.log_figure("Feature Importances", fig)

    return test_accuracy_list[-1]
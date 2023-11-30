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
# self written
from utils.plots_classifier import plot_loss_function, plot_data
from utils.models import SimpleNN
# logging
from utils.log import setup_comet_logger
import logging
logger = logging.getLogger(__name__)



# ************* Create custom Dataset ************* #
class CustomDataset(Dataset):
    def __init__(self, data, labels, pipelines=None, device=None):
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
        self.pipelines = pipelines
        self.all_variables = ["probe_pt", "probe_eta", "probe_phi", "probe_fixedGridRhoAll", "probe_r9", "probe_s4",
                              "probe_sieie", "probe_sieip", "probe_etaWidth", "probe_phiWidth", "probe_pfPhoIso03",
                              "probe_pfChargedIsoPFPV", "probe_pfChargedIsoWorstVtx", "probe_energyRaw"]
        
        if self.pipelines is not None:
            for var, pipeline in self.pipelines.items():
                if var in self.all_variables:
                    trans = (
                        pipeline.transform
                    )
                    data[var] = trans(data[var].values.reshape(-1, 1)).reshape(-1)

        self.data = torch.tensor(data.values, dtype=torch.float32, device=device)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# ********* Create training and test data ********* #
def create_data(device, cfg):

    if cfg.data.name == "sonar":
        sonar_data = pd.read_csv("/work/tfluehma/git/SSNF2/classifier/data/sonar_data.csv")
        X = sonar_data.iloc[:, 0:60]
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

    else:
        train_data = pd.read_parquet("/work/tfluehma/git/SSNF2/classifier/data/train_data.parquet")
        test_data = pd.read_parquet("/work/tfluehma/git/SSNF2/classifier/data/test_data.parquet")

        train = train_data.iloc[:, :-1]
        train_label = train_data["label"]

        test = test_data.iloc[:, :-1]
        test_label = test_data["label"]

        # Change string labels to integers
        encoder = LabelEncoder()
        encoder.fit(train_label)
        train_label = encoder.transform(train_label)
        test_label = encoder.transform(test_label)

        # Define the pipelines
        with open("/work/tfluehma/git/SSNF2/preprocess/pipelines_eb.pkl", "rb") as file:
            pipelines = pkl.load(file)
            pipelines = pipelines["pipe1"]

        # Create Dataset
        train_dataset = CustomDataset(train, train_label, pipelines=pipelines, device=device)
        test_dataset = CustomDataset(test, test_label, pipelines=pipelines, device=device)

        # Plot the data to verify preprocessing worked
        plot_data(train_dataset.data, keys=train.keys())

        # Create Dataloader
        batch_size = cfg.hyperparameters.batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader

# **************** Train Function ***************** #
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()
    epoch_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch % 5000 == 0:
            testing_loss, current = loss.item(), (batch + 1) * len(X)
            correct = (pred.round() == y).float().mean()
            logger.info(f"current batch loss: {testing_loss:>7f} Accuracy: {(100*correct):>0.1f}%  [{current:>5d}/{size:>5d}]")
    return epoch_loss

# ***************** Test Function ***************** #
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluation the model with torch.no_grad() ensures that no gradients
    # are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).float().sum()

    test_loss
    correct /= size
    logger.info(f"Complete Test loss: {test_loss:>8f}, Total Accuracy: {(100*correct):>0.1f}% \n")
    
    return test_loss, 100*correct

# ******** classifier function containing training, testing and keeping track of results ********
def classify(device, cfg):
    # create the datasets
    train_dataloader, test_dataloader = create_data(device, cfg)
    # create model
    input_size = train_dataloader.dataset.data.shape[1]
    num_layers = cfg.model.num_layers
    hidden_size = cfg.model.hidden_size
    model = SimpleNN(input_size, hidden_size, num_layers).to(device)
    logger.info("Training with Model: \n{}".format(model))
    # Define Hyperparameters
    learning_rate = cfg.hyperparameters.learning_rate
    epochs = 100

    # initialize the loss function and optimizer
    loss_fn = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup Comet logger
    if cfg.logger:
        comet_name = os.getcwd().split("/")[-1]
        comet_logger = setup_comet_logger(comet_name, cfg.model)

    # **** Train and Test ****
    # keep track of models loss
    train_loss_list = []
    test_loss_list = []
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn)
        if cfg.logger:
            comet_logger.log_metrics({"epoch_train_loss": train_loss, "total_test_loss": test_loss,
                                       "test_accuracy": test_accuracy}, step=t)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    plot_loss_function(training_loss=train_loss_list, testing_loss=test_loss_list)
        


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import copy

# Get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# ************* Create custom Dataset ************* #
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values, dtype=torch.float32, device=device)
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# Create training and test data
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

# Create Dataset
train_dataset = CustomDataset(train, train_label)
test_dataset = CustomDataset(test, test_label)

# Create Dataloader
batch_size=512
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# ************** Build the Neural Network ******************
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
    
model = NeuralNetwork().to(device)
print(model)

# Define Hyperparameters
learning_rate = 1e-3
epochs = 5




# **************** Train Function **************** #
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            correct = (pred.round() == y).float().mean()
            print(f"loss: {loss:>7f} Accuracy: {(100*correct):>0.1f}%  [{current:>5d}/{size:>5d}]")
            

# **************** Test Function **************** #
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Hold the best model
    best_acc = -np.inf
    best_weights = None

    # Evaluation the model with torch.no_grad() ensures that no gradients
    # are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # TODO: this correct is not fit for this model. Correct it
            correct += (pred.round() == y).float().mean()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if correct > best_acc:
                best_acc = correct
                best_weights = copy.deepcopy(model.state_dict())
                print("new best model")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
# initialize the loss function and optimizer
loss_fn = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ********************* Actual Testing ********************* #
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done")

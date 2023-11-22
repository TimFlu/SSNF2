import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import os

# ************* Create custom Dataset ************* #
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.data.iloc[index]
        y = self.labels.iloc[index]
        return x, y

# Create training and test data
train_data = pd.read_parquet("SSNF2/classifier/data/train_data.parquet")
test_data = pd.read_parquet("SSNF2/classifier/data/test_data.parquet")

train = train_data.iloc[:, :-1]
train_label = train_data["label"]

test = test_data.iloc[:, :-1]
test_label = test_data["label"]

train_dataset = CustomDataset(train, train_label)
test_dataset = CustomDataset(test, test_label)

# Create Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

print(len(train_dataloader))

# ************** Build the Neural Network ******************
# Get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear()
        )
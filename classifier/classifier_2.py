import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import os

# ************* Create custom Dataset ************* #
class CustomDataset(Dataset):
    def __init__(self, labels, data_dir):
        self.data_dir = data_dir
        self.img.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return super().__getitem__(index)

# Create training and test data
data_eb_test = pd.read_parquet("SSNF2/classifier/data/data_eb_test_classifier.parquet")
data_eb_train = pd.read_parquet("SSNF2/classifier/data/data_eb_train_classifier.parquet")
mc_eb_test = pd.read_parquet("SSNF2/classifier/data/mc_eb_test_classifier.parquet")
mc_eb_train = pd.read_parquet("SSNF2/classifier/data/mc_eb_train_classifier.parquet")
print(data_eb_test.shape[1] == data_eb_train.shape[1])

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

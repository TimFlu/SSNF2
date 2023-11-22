import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LabelEncoder used to convert string label to an integer ** not sure if used **
from sklearn.preprocessing import LabelEncoder

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import copy


#******* Prepare the data *******#
data_eb_train = pd.read_parquet("SSNF2/preprocess/data_eb_train.parquet")
# print(data_eb_train.head())
# print(data_eb_train.keys())
# print("shape: ", data_eb_train.shape)
mc_eb_train = pd.read_parquet("SSNF2/preprocess/mc_eb_train.parquet")
# print(mc_eb_train.head())
# print("shape: ", mc_eb_train.shape)

# Create a test dataset just for probe_pt
data_eb_train["label"] = data_eb_train.shape[0] * ["data"]
data_probe_pt = data_eb_train[["probe_pt", "label"]]

mc_eb_train["label"] = mc_eb_train.shape[0] * ["mc"]
mc_probe_pt = mc_eb_train[["probe_pt", "label"]]

probe_pt = pd.concat([data_probe_pt, mc_probe_pt], axis=0)
probe_pt_train = probe_pt.iloc[:, :-1]
label = probe_pt.iloc[:, -1]


# Change string labels to integers mc -> 1, data -> 0
encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label)


# Convert the data into pytorch tensors
probe_pt_train = torch.tensor(probe_pt_train.values, dtype=torch.float64)
label = torch.tensor(label, dtype=torch.float64).reshape(-1, 1)
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    probe_pt_train = probe_pt_train.to("cuda")


#******* Create a Model *******#
size = len(probe_pt_train)
print("shape: ", probe_pt_train.shape)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(size, 5)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(5, 5)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(5, 5)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
print(sum([x.reshape(-1).shape[0] for x in Classifier().parameters()]))

        
#********* Training function ********#

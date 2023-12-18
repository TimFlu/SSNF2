
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.distributed import init_process_group
import os
import numpy as np
import pandas as pd
from copy import deepcopy


class CustomDataset(Dataset):
    def __init__(self, data, labels, pipelines=None, target_only=False, device=None):
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
        self.pipelines = pipelines
        self.context_variables = ["probe_pt", "probe_eta", "probe_phi", "probe_fixedGridRhoAll"]
        self.all_variables = ["probe_pt", "probe_eta", "probe_phi", "probe_fixedGridRhoAll", "probe_r9", "probe_s4",
                              "probe_sieie", "probe_sieip", "probe_etaWidth", "probe_phiWidth", "probe_pfPhoIso03",
                              "probe_pfChargedIsoPFPV", "probe_pfChargedIsoWorstVtx", "probe_energyRaw"]
        
        if 'weight' in self.data.columns:
            # delete weights in data and store them seperately as we do not want to
            # train the classifier on the weights
            self.weight = torch.tensor(data["weight"].values, dtype=torch.float32, device=device)
            del self.data["weight"]
        
        if self.pipelines is not None:
            for var, pipeline in self.pipelines.items():
                if var in self.all_variables:
                    trans = (
                        pipeline.transform
                    )
                    data[var] = trans(data[var].values.reshape(-1, 1)).reshape(-1)
        
        if target_only:
            for context in self.context_variables:
                del self.data[context]

        # create tensor from DataFrame
        self.data = torch.tensor(data.values, dtype=torch.float32, device=device)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        weight = self.weight[index]
        return x, y, weight



class ParquetDataset(Dataset):
    def __init__(
        self,
        parquet_file,
        context_variables,
        target_variables,
        device=None,
        pipelines=None,
        retrain_pipelines=False,
        rows=None,
    ):
        self.parquet_file = parquet_file
        self.context_variables = context_variables
        self.target_variables = target_variables
        self.all_variables = context_variables + target_variables
        data = pd.read_parquet(
            parquet_file, columns=self.all_variables + ["weight", "probe_energyRaw"], engine="fastparquet"
        )
        self.pipelines = pipelines
        if self.pipelines is not None:
            for var, pipeline in self.pipelines.items():
                if var in self.all_variables:
                    trans = (
                        pipeline.fit_transform
                        if retrain_pipelines
                        else pipeline.transform
                    )
                    data[var] = trans(data[var].values.reshape(-1, 1)).reshape(-1)
        if rows is not None:
            data = data.iloc[:rows]
        self.target = data[target_variables].values
        self.context = data[context_variables].values
        self.weight = data["weight"].values
        self.extra = data["probe_energyRaw"].values
        if device is not None:
            self.target = torch.tensor(self.target, dtype=torch.float32).to(device)
            self.context = torch.tensor(self.context, dtype=torch.float32).to(device)
            self.weight = torch.tensor(self.weight, dtype=torch.float32).to(device)
            self.extra = torch.tensor(self.extra, dtype=torch.float32).to(device)

    def __len__(self):
        assert len(self.context) == len(self.target)
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx], self.weight[idx], self.extra[idx]

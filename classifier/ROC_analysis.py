import pandas as pd
import hydra
from utils.models import SimpleNN, get_zuko_nsf, load_fff_model
from utils.datasets import ParquetDataset, CustomDataset
from sklearn.metrics import roc_curve
import torch
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the config
path_to_output_folder = "/work/tfluehma/git/outputs/classifier_test_config-2023-12-13-11-32-13"
config_name = "run_classifier"
os.chdir(path_to_output_folder)

@hydra.main(version_base=None, config_path=path_to_output_folder, config_name=config_name)
def main(cfg):

    # load best fff model
    flow_params_dct = {
            "input_dim": 9,
            "context_dim": 4,
            "ntransforms": cfg.fff_model.ntransforms,
            "nbins": cfg.fff_model.nbins,
            "nnodes": cfg.fff_model.nnodes,
            "nlayers": cfg.fff_model.nlayers,
        }
    penalty = {
        "penalty_type": cfg.fff_model.penalty,
        "penalty_weight": cfg.fff_model.penalty_weight,
        "anneal": cfg.fff_model.anneal,
    }
    model_fff = get_zuko_nsf(**flow_params_dct)
    model_fff, _, _, _, _, _ = load_fff_model(
        top_file=cfg.checkpoints.top, mc_file=cfg.checkpoints.mc, data_file=cfg.checkpoints.data,
        top_penalty=penalty
    )
    model_fff.to(device)
    
    # Define the pipelines
    with open("/work/tfluehma/git/SSNF2/preprocess/pipelines_eb.pkl", "rb") as file:
        pipelines_data = pkl.load(file)
        pipelines_data = pipelines_data["pipe1"]
    
    # Read in mc test set
    test_file_mc = f"{path_to_output_folder}/../../SSNF2/preprocess/mc_eb_test.parquet"

    test_dataset_mc_full = ParquetDataset(
        test_file_mc,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=None,
    )
    test_loader_mc_full = DataLoader(
            test_dataset_mc_full,
            batch_size=2048,
            shuffle=False,
        )
    
    mc_list, mc_corrected_list = [], []
    mc_context_list, mc_corrected_context_list = [], []
    mc_extra_list = []
    with torch.no_grad():
        for mc in test_loader_mc_full:
            context_mc, target_mc, weights_mc, extra_mc = mc
            target_mc_corr, _ = model_fff.transform(
                    target_mc, context_mc, inverse=False
                )
            target_mc = target_mc.detach().cpu().numpy()
            target_mc_corr = target_mc_corr.detach().cpu().numpy()
            context_mc = context_mc.detach().cpu().numpy()
            mc_list.append(target_mc)
            mc_corrected_list.append(target_mc_corr)
            mc_context_list.append(context_mc)
            mc_corrected_context_list.append(context_mc)
            mc_extra_list.append(extra_mc)
            
    mc_target = np.concatenate(mc_list, axis=0)
    mc_target_corr = np.concatenate(mc_corrected_list, axis=0)
    mc_context = np.concatenate(mc_context_list, axis=0)
    mc_extra = np.concatenate(mc_extra_list, axis=0).reshape(-1, 1)
    
    # Concat target with context and extra
    mc_target_context = np.concatenate((mc_target, mc_context, mc_extra), axis=1)
    mc_target_corr_context = np.concatenate((mc_target_corr, mc_context, mc_extra), axis=1)

    mc_target_context = pd.DataFrame(mc_target_context, columns=cfg.target_variables+cfg.context_variables+["extra"])
    mc_target_corr_context = pd.DataFrame(mc_target_corr_context, columns=cfg.target_variables+cfg.context_variables+["extra"])
    
    # Create the CustomDataset and DataLoader to bring the data in the right format
    # for the classifier
    mc_dataset = CustomDataset(data=mc_target_context, labels=np.ones(len(mc_target_context)), device=device)
    mc_corr_dataset = CustomDataset(data=mc_target_corr_context, labels=np.ones(len(mc_target_corr_context)), device=device)
    mc_target_dataloader = DataLoader(mc_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
    mc_target_dataloader = DataLoader(mc_corr_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
    
    # create and load classifier model with best weights
    input_size = len(cfg.context_variables) + len(cfg.target_variables) + 1
    num_layers = cfg.model.num_layers
    hidden_size = cfg.model.hidden_size
    model_classifier = SimpleNN(input_size, hidden_size, num_layers).to(device)
    model_classifier.load_state_dict(torch.load("./best_model_weights.pth"))

    # plot ROC curve
    params_label = f"layers: {cfg.model.num_layers}, nodes: {cfg.model.hidden_size},\
            batch_size = {cfg.hyperparameters.batch_size}, LR = {cfg.hyperparameters.learning_rate}"
    with torch.no_grad():
        plt.figure()
        y_test = mc_target_dataloader.dataset.labels.to("cpu")
        X_test = mc_target_dataloader.dataset.data
        y_pred = model_classifier(X_test)
        y_pred = y_pred.to("cpu")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label=params_label)
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig("./plots//ROC_test")












if __name__ == "__main__":
    main()
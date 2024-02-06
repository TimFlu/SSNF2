import pandas as pd
import hydra
from utils.models import SimpleNN, get_zuko_nsf, load_fff_model
from utils.datasets import ParquetDataset, CustomDataset
from utils.phoid import calculate_photonid_mva
from utils.plots_classifier import plot_data, dump_main_plot, dump_full_profile_plot, feature_importance
from utils.plots_classifier import classifier_corrected_mc_plot, transformed_ranges
from sklearn.metrics import roc_curve
import torch
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# load the config
path_to_output_folder = "/work/tfluehma/git/SSNF2/outputs/classifier_HL10HS60LR4-2023-12-21-11-51-43"
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
    
    # Read in mc and data test set  
    test_file_mc = f"{path_to_output_folder}/../../classifier/data/balanced_test_mc.parquet"
    test_file_data = f"{path_to_output_folder}/../../classifier/data/balanced_test_data.parquet"
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
            shuffle=True,
        )
    test_dataset_data_full = ParquetDataset(
        test_file_data,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=None
        )
    
    # preprocess data and create DataFrame
    context_data, target_data, weights_data, extra_data = test_dataset_data_full[:]
    test_data = np.concatenate((context_data.detach().cpu().numpy(),
                                target_data.detach().cpu().numpy(),
                                extra_data.detach().cpu().numpy().reshape(-1, 1)), axis=1)
    test_data = pd.DataFrame(test_data, columns=cfg.context_variables+cfg.target_variables+["probe_energyRaw"])
    plot_data(test_data, test_data.keys(), name="_testing_data_preprocess")

    # correct MC
    mc_list, mc_corrected_list = [], []
    mc_context_list, mc_corrected_context_list = [], []
    mc_weights_lst = []
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
            weights_mc = weights_mc.detach().cpu().numpy()
            mc_list.append(target_mc)
            mc_corrected_list.append(target_mc_corr)
            mc_context_list.append(context_mc)
            mc_corrected_context_list.append(context_mc)
            mc_extra_list.append(extra_mc.detach().cpu().numpy())
            mc_weights_lst.append(weights_mc)
            if len(mc_list) % 10 == 0:
                print("transformed", len(mc_list), "batches out of", len(test_loader_mc_full))
            
    mc_target = np.concatenate(mc_list, axis=0)
    mc_target_corr = np.concatenate(mc_corrected_list, axis=0)
    mc_context = np.concatenate(mc_context_list, axis=0)
    mc_extra = np.concatenate(mc_extra_list, axis=0).reshape(-1, 1)
    weights_mc = np.concatenate(mc_weights_lst, axis=0)

    # Concat target with context and probe_energyRaw
    mc_target_context = np.concatenate((mc_context, mc_target, mc_extra), axis=1)
    mc_target_corr_context = np.concatenate((mc_context, mc_target_corr, mc_extra), axis=1)

    mc_target_context = pd.DataFrame(mc_target_context, columns=cfg.context_variables+cfg.target_variables+["probe_energyRaw"])
    mc_target_corr_context = pd.DataFrame(mc_target_corr_context, columns=cfg.context_variables+cfg.target_variables+["probe_energyRaw"])
    
    # plot the corrected MC and uncorrected MC to verify that the model corrected them as supposed
    # Create copies
    data_df_mva = test_data.copy()
    mc_df_mva = mc_target_context.copy()
    mc_corr_df_mva = mc_target_corr_context.copy()
    classifier_corrected_mc_plot(data_df_mva, mc_df_mva, mc_corr_df_mva, weights_mc,
                                 pipelines_data, path_to_output_folder, cfg)
    
    # Concat MC with data
    mc_and_data = pd.concat([mc_target_context, test_data], axis=0)
    mc_corr_and_data = pd.concat([mc_target_corr_context, test_data], axis=0)

    # plot the data to verify the distributions
    plot_data(mc_and_data, mc_and_data.keys(), name="_mc_and_data")
    plot_data(mc_corr_and_data, mc_corr_and_data.keys(), name="_mc_corr_and_data")

    # Create the CustomDataset and DataLoader to bring the data in the right format
    # for the classifier
    labels = [1 for _ in range(len(mc_target_context))] + [0 for _ in range(len(test_data))]
    mc_and_data_dataset = CustomDataset(data=mc_and_data, labels=labels, target_only=cfg.data.target_only, device=device)
    mc_corr_and_data_dataset = CustomDataset(data=mc_corr_and_data, labels=labels, target_only=cfg.data.target_only, device=device)
    mc_and_data_dataloader = DataLoader(mc_and_data_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    mc_corr_and_data_dataloader = DataLoader(mc_corr_and_data_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    
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

        # uncorrected MC
        y_test = mc_and_data_dataloader.dataset.labels.to("cpu")
        X_test = mc_and_data_dataloader.dataset.data
        y_pred = model_classifier(X_test)
        y_pred = y_pred.to("cpu")
        accuracy = ((y_pred.round() == y_test).float().sum()/len(y_pred)).cpu().numpy()
        print("Accuracy uncorr: ", accuracy*100,"%")

        # corrected MC
        y_test_corr = mc_corr_and_data_dataloader.dataset.labels.to("cpu")
        X_test_corr = mc_corr_and_data_dataloader.dataset.data
        y_pred_corr = model_classifier(X_test_corr)
        y_pred_corr = y_pred_corr.to("cpu")
        accuracy_corr = ((y_pred_corr.round() == y_test_corr).float().sum()/len(y_pred_corr)).cpu().numpy()
        print("Accuracy corr: ", accuracy_corr*100,"%")


        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        ax[0].plot(fpr, tpr, label=params_label + f", acc = {np.round(accuracy*100, 2)}%")
        ax[0].set_title("Receiver Operating Characteristics with uncorrected MC")
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].legend()

        fpr_corr, tpr_corr, thresholds = roc_curve(y_test_corr, y_pred_corr)
        ax[1].plot(fpr_corr, tpr_corr, label=params_label + f", acc = {np.round(accuracy_corr*100, decimals=2)}%")
        ax[1].set_title("Receiver Operating Characteristics with corrected MC")
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig("./plots/ROC2.png")



    # plot feature importance
    feature_importance(model_classifier, mc_and_data_dataloader.dataset.data, cfg, device)
    feature_importance(model_classifier, mc_corr_and_data_dataloader.dataset.data, cfg, device, corrected=True)
if __name__ == "__main__":
    main()
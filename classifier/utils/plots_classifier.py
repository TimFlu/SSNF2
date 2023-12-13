import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import torch
import os
from utils.models import SimpleNN, get_zuko_nsf, load_fff_model



def plot_loss_function(training_loss, testing_loss):
    # create plots folder if it does not exist
    folder_name = os.getcwd() + "/plots/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created succesfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    

    # plot loss function
    fig, ax = plt.subplots(2)
    x_ax = np.linspace(0, len(training_loss), len(training_loss))
    ax[0].plot(x_ax, training_loss, label="training_loss")
    ax[1].plot(x_ax, testing_loss, label="testing_loss")
    ax[0].legend()
    ax[1].legend()
    
    plt.savefig(folder_name + "/loss")

def plot_data(data, keys):
    # create plots folder if it does not exist
    folder_name = os.getcwd() + "/plots/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created succesfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

    # plot preprocessed data
    data_ = data
    data_ = data_.to("cpu")
    data_ = pd.DataFrame(data_)
    fig, ax = plt.subplots(len(keys), figsize=(6,22))
    for i, key in enumerate(keys):
        ax[i].hist(data_.iloc[:, i], bins=100, label=key)
        ax[i].legend(loc="best")
    plt.savefig(folder_name + "/preprocessed_data")

def roc_plot(train_dataloader, cfg, device):
    # create plots folder if it does not exist
    folder_name = os.getcwd() + "/plots/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created succesfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    # create and load model with best weights
    input_size = train_dataloader.dataset.data.shape[1]
    num_layers = cfg.model.num_layers
    hidden_size = cfg.model.hidden_size
    model = SimpleNN(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load("./best_model_weights.pth"))
    # plot ROC curve
    params_label = f"layers: {cfg.model.num_layers}, nodes: {cfg.model.hidden_size},\
          batch_size = {cfg.hyperparameters.batch_size}, LR = {cfg.hyperparameters.learning_rate}"
    with torch.no_grad():
        plt.figure()
        y_test = train_dataloader.dataset.labels.to("cpu")
        X_test = train_dataloader.dataset.data
        y_pred = model(X_test)
        y_pred = y_pred.to("cpu")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label=params_label)
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(folder_name + "/ROC")


def roc_plot_corrected_mc(mc_uncorr_dataloader, cfg, device):
    # load the corrected fff model
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

    with torch.no_grad():
        for mc in mc_uncorr_dataloader:
            context_mc, target_mc = mc
            target_mc_corr, _ = model_fff.transform(
                    target_mc, context_mc, inverse=False
                )
    print(target_mc)
    print(target_mc_corr)
    print(target_mc_corr == target_mc)

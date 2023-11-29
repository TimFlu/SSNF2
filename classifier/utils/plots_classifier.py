import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_loss_function(training_loss, testing_loss):
    fig, ax = plt.subplots(2)
    x_ax = np.linspace(0, len(training_loss), len(training_loss))
    ax[0].plot(x_ax, training_loss, label="training_loss")
    ax[1].plot(x_ax, testing_loss, label="testing_loss")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("/work/tfluehma/git/SSNF2/classifier/data/loss")

def plot_data(data, keys):
    data_ = data
    data_ = data_.to("cpu")
    data_ = pd.DataFrame(data_)
    fig, ax = plt.subplots(len(keys), figsize=(6,22))
    for i, key in enumerate(keys):
        ax[i].hist(data_.iloc[:, i], bins=100, label=key)
        ax[i].legend(loc="best")
    plt.savefig("/work/tfluehma/git/SSNF2/classifier/data/preprocessed")

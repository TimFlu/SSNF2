import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import torch
import os
import mplhep as hep
hep.style.use("CMS")
import logging
import json
from utils.models import SimpleNN, get_zuko_nsf, load_fff_model

logger = logging.getLogger(__name__)
from pathlib import Path

from utils.phoid import calculate_photonid_mva

script_dir = Path(__file__).parent.absolute()

transformed_ranges = {
    "pipe0": {
        "probe_pt": [-4, 4],
        "probe_eta": [-2, 2],
        "probe_phi": [-2, 2],
        "probe_fixedGridRhoAll": [-3, 5],
        "probe_r9": [-2, 2],
        "probe_s4": [-2, 3],
        "probe_sieie": [-6, 6],
        "probe_sieip": [-6, 6],
        "probe_etaWidth": [-3, 5],
        "probe_phiWidth": [-3, 3],
        "probe_pfPhoIso03": [-4, 3],
        "probe_pfChargedIsoPFPV": [-4, 3],
        "probe_pfChargedIsoWorstVtx": [-3, 6],
        "probe_energyRaw": [0, 300],
    },
    "pipe1": {
        "probe_pt": [-4, 4],
        "probe_eta": [-2, 2],
        "probe_phi": [-2, 2],
        "probe_fixedGridRhoAll": [-3, 5],
        "probe_r9": [-2, 2],
        "probe_s4": [-2, 3],
        "probe_sieie": [-6, 6],
        "probe_sieip": [-6, 6],
        "probe_etaWidth": [-3, 5],
        "probe_phiWidth": [-3, 3],
        "probe_pfPhoIso03": [-3, 3],
        "probe_pfChargedIsoPFPV": [-2, 3.5],
        "probe_pfChargedIsoWorstVtx": [-5, 6],
        "probe_energyRaw": [0, 300],
    },
    "pipe_cqmnf1": {
        "probe_pt": [-4, 4],
        "probe_eta": [-2, 2],
        "probe_phi": [-2, 2],
        "probe_fixedGridRhoAll": [-3, 5],
        "probe_r9": [-2, 2],
        "probe_s4": [-2, 3],
        "probe_sieie": [-6, 6],
        "probe_sieip": [-6, 6],
        "probe_etaWidth": [-3, 5],
        "probe_phiWidth": [-3, 3],
        "probe_pfPhoIso03": [-3, 3],
        "probe_pfChargedIsoPFPV": [-2, 3.5],
        "probe_pfChargedIsoWorstVtx": [-5, 6],
        "probe_energyRaw": [0, 300],
    }
}


def classifier_corrected_mc_plot(data_df_mva, mc_df_mva, mc_corr_df_mva, 
                                 weights_mc, pipelines_data,
                                 path_to_output_folder, cfg, comet_logger):
    
    # create folder if it does not exist
    folder_name = os.getcwd() + "/plots/transform_analysis/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created succesfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    # plot the corrected MC and uncorrected MC to verify that the model corrected them as supposed
    for var in cfg.target_variables:
        dump_main_plot(
            data_df_mva[var],
            mc_df_mva[var],
            variable_conf={
                "name": var,
                "title": var,
                "x_label": var,
                "bins": 100,
                "range": transformed_ranges["pipe1"][var],
            },
            output_dir="./plots/transform_analysis/",
            subdetector="eb",
            mc_corr=mc_corr_df_mva[var],
            weights=weights_mc,
            extra_name=f"_top_transformed"
        )    
    
    # sample back
    # note that pipelines are actually the same, trained on data
    data_pipeline = pipelines_data
    mc_pipeline = pipelines_data

    with open(f"{path_to_output_folder}/../../preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        vars_config = {d["name"]: d for d in vars_config}

    for var in cfg.target_variables:
        data_df_mva[var] = (
            data_pipeline[var]
            .inverse_transform(data_df_mva[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_df_mva[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_df_mva[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_corr_df_mva[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_corr_df_mva[var].values.reshape(-1, 1))
            .reshape(-1)
        )

    for var in cfg.context_variables:
        data_df_mva[var] = (
            data_pipeline[var]
            .inverse_transform(data_df_mva[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_df_mva[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_df_mva[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_corr_df_mva[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_corr_df_mva[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    
    data_df_mva["probe_mvaID"] = calculate_photonid_mva(data_df_mva, calo="eb")
    mc_df_mva["probe_mvaID"] = calculate_photonid_mva(mc_df_mva, calo="eb")
    mc_corr_df_mva["probe_mvaID"] = calculate_photonid_mva(mc_corr_df_mva, calo="eb")

    for var in cfg.target_variables + ["probe_mvaID"]:
            dump_main_plot(
                data_df_mva[var],
                mc_df_mva[var],
                variable_conf=vars_config[var],
                output_dir="./plots/transform_analysis/",
                subdetector="eb",
                mc_corr=mc_corr_df_mva[var],
                weights=weights_mc,
                extra_name="_top"
            )

    # now plot profiles
    nbins = 8
    for column in cfg.target_variables + ["probe_mvaID"]:
        for cond_column in cfg.context_variables:
            dump_full_profile_plot(
                nbins,
                column,
                cond_column,
                data_df_mva,
                mc_df_mva,
                mc_corr_df_mva,
                subdetector="eb",
                weights=weights_mc,
                output_dir="./plots/transform_analysis/",
                extra_name="_top",
                cometlogger_epoch=[comet_logger, 0]
            )


def feature_importance(model, X_data, cfg, comet_logger, device, name="", corrected=False):
    if corrected:
        savepath = f"./plots/{name}_feature_importance_corrected.png"
    else:
        savepath = f"./plots/{name}_feature_importance.png"
    if cfg.data.target_only:
        label = cfg.target_variables + ["probe_energyRaw"]
    else:
        label = cfg.context_variables + cfg.target_variables + ["probe_energyRaw"]
    model.eval()
    X_tensor = torch.tensor(X_data, dtype=torch.float32, requires_grad=True, device=device)
    output = model(X_tensor)
    output.backward(torch.ones_like(output))
    feature_importance = X_tensor.grad.abs().mean(dim=0).cpu()
    # Plotting feature importances
    fig = plt.figure(figsize=(12, 12))
    plt.bar(range(len(feature_importance)), feature_importance, align='center')
    if cfg.data.name != "sonar":
        plt.xticks(ticks=range(len(feature_importance)), labels=label, rotation='vertical', fontsize=9)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.title('Feature Importances')
    plt.savefig(savepath)
    if cfg.logger:
        comet_logger.log_figure(f"{name} Feature Importances", fig)


def plot_loss_function(training_loss, testing_loss, comet_logger, cfg):
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
    if cfg.logger:
        comet_logger.log_figure("loss", fig)


def plot_data(data, keys, comet_logger, cfg, name=None):
    """
    Samples the data and returns a plot.

    Parameters: 
    data (DataFrame / Tensor): data wished to sample.
    keys: column names of the data.
    name (str, optinal): name to save the plot as.
    """
    # create plots folder if it does not exist
    folder_name = os.getcwd() + "/plots/"
    if name is not None:
        comet_name = "preprocessed_data" + name
    else:
        comet_name = "preprocessed_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created succesfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

    # plot preprocessed data
    if isinstance(data, pd.DataFrame):
        data_ = data
    else:
        data_ = data
        data_ = data_.to("cpu")
        data_ = pd.DataFrame(data_)
        
    fig, ax = plt.subplots(len(keys), figsize=(6,22))
    for i, key in enumerate(keys):
        ax[i].hist(data_.iloc[:, i], bins=100, label=key)
        ax[i].legend(loc="best")
    plt.savefig(folder_name + "/preprocessed_data" + name)
    if cfg.logger:
        comet_logger.log_figure(comet_name, fig)


def roc_plot(train_dataloader, cfg, comet_logger, device):
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
        fig = plt.figure(figsize=(16,16))
        y_test = train_dataloader.dataset.labels.to("cpu")
        X_test = train_dataloader.dataset.data
        y_pred = model(X_test)
        y_pred = y_pred.to("cpu")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=params_label + f" AUC={roc_auc}")
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(folder_name + "/ROC")
        
        if cfg.logger:
            comet_logger.log_figure("Receiver Operatin Characteristics", fig)
            comet_logger.log_metrics({"AUC": roc_auc})


def divide_dist(distribution, bins):
    sorted_dist = np.sort(distribution)
    subgroup_size = len(distribution) // bins
    edges = [sorted_dist[0]]
    for i in range(subgroup_size, len(sorted_dist), subgroup_size):
        edges.append(sorted_dist[i])
    edges[-1] = sorted_dist[-1]
    return edges


def interpolate_weighted_quantiles(values, weights, quantiles):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]


def dump_profile_plot(
    ax, ss_name, cond_name, sample_name, ss_arr, cond_arr, color, cond_edges, weights
):
    df = pd.DataFrame({ss_name: ss_arr, cond_name: cond_arr, "weights": weights})
    quantiles = [0.25, 0.5, 0.75]
    qlists = [[], [], []]
    centers = []
    for left_edge, right_edge in zip(cond_edges[:-1], cond_edges[1:]):
        dff = df[(df[cond_name] > left_edge) & (df[cond_name] < right_edge)]
        # procedure for weighted quantiles
        data = dff[ss_name].values
        weights = dff["weights"].values
        qlist = interpolate_weighted_quantiles(data, weights, quantiles)
        for i, q in enumerate(qlist):
            qlists[i].append(q)
        centers.append((left_edge + right_edge) / 2)
    mid_index = len(quantiles) // 2
    for qlist in qlists[:mid_index]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    for qlist in qlists[mid_index:]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    ax.plot(centers, qlists[mid_index], color=color, label=sample_name)

    return ax


def dump_full_profile_plot(
    nbins,
    target_variable,
    cond_variable,
    data_df,
    mc_uncorr_df,
    mc_corr_df,
    subdetector,
    weights,
    output_dir="",
    extra_name="",
    writer_epoch=None,
    cometlogger_epoch=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    data_ss_arr = data_df[target_variable].values
    data_cond_arr = data_df[cond_variable].values
    mc_uncorr_ss_arr = mc_uncorr_df[target_variable].values
    mc_uncorr_cond_arr = mc_uncorr_df[cond_variable].values
    mc_corr_ss_arr = mc_corr_df[target_variable].values
    mc_corr_cond_arr = mc_corr_df[cond_variable].values
    cond_edges = divide_dist(data_cond_arr, nbins)

    for name, ss_arr, cond_arr, color, w in [
        ("data", data_ss_arr, data_cond_arr, "black", np.ones(len(data_ss_arr))),
        ("mc", mc_uncorr_ss_arr, mc_uncorr_cond_arr, "red", weights),
        ("mc corr", mc_corr_ss_arr, mc_corr_cond_arr, "blue", weights),
    ]:
        ax = dump_profile_plot(
            ax=ax,
            ss_name=target_variable,
            cond_name=cond_variable,
            sample_name=name,
            ss_arr=ss_arr,
            cond_arr=cond_arr,
            color=color,
            cond_edges=cond_edges,
            weights=w,
        )
    ax.legend()
    ax.set_xlabel(cond_variable)
    ax.set_ylabel(target_variable)
    # reduce dimension of labels and axes names
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    fig.tight_layout()
    fig_name = f"profiles_{target_variable}_{cond_variable}_{subdetector}{extra_name}"

    if writer_epoch is not None:
        writer, epoch = writer_epoch
        writer.add_figure(fig_name, fig, epoch)
    if cometlogger_epoch is not None:
        comet_logger, epoch = cometlogger_epoch
        comet_logger.log_figure(fig_name, fig, step=epoch)
    if writer_epoch is None and cometlogger_epoch is None:
        if type(output_dir) == str:
            output_dir = [output_dir]
        for dr in output_dir:
            for ext in ["pdf", "png"]:
                fig.savefig(dr + "/" + fig_name + "." + ext, bbox_inches="tight")
    plt.close(fig)


def print_mc_hist(
    up, down, bins, range, label, data_hist, data_hist_norm, data_centers, data_err, mc, color, weights
):
    if weights is None:
        weights = np.ones(len(mc))

    mc_hist, mc_bins = np.histogram(
        mc, bins=bins, range=range, weights=weights
    )
    mc_hist_norm, _, _ = up.hist(
        mc, bins=bins, range=range, weights=weights, density=True, label=label, color=color, histtype="step"
    )
    mc_err = np.sqrt(np.histogram(mc, bins=bins, range=range, weights=weights**2)[0])
    mc_err_norm = mc_err / (np.diff(mc_bins) * np.sum(weights))
    up.errorbar(
        data_centers,
        mc_hist_norm,
        yerr=mc_err_norm,
        color=color,
        marker="",
        linestyle="",
        markersize=4,
    )

    rdatamc_hist = data_hist_norm / mc_hist_norm
    rdatamc_err = (
        np.sqrt((data_err / data_hist) ** 2 + (mc_err / mc_hist) ** 2)
    ) * rdatamc_hist
    down.errorbar(
        data_centers,
        rdatamc_hist,
        yerr=rdatamc_err,
        color=color,
        marker="o",
        linestyle="",
        markersize=4,
    )

    return up, down


def dump_main_plot(
    data,
    mc_uncorr,
    variable_conf,
    output_dir,
    subdetector,
    mc_corr=None,
    weights=None,
    extra_name="",
    labels=None,
    writer_epoch=None,
    cometlogger_epoch=None,
):
    name = variable_conf["name"]
    title = variable_conf["title"] + "_" + subdetector
    x_label = variable_conf["x_label"]
    bins = variable_conf["bins"]
    range = variable_conf["range"]

    if type(output_dir) == str:
        output_dir = [output_dir]

    # specific ranges for EB and EE
    if name == "probe_sieie" and subdetector == "EE":
        range = [0.005, 0.04]

    if labels is None:
        labels = [
            f"Data - {subdetector}",
            f"MC - {subdetector} (uncorr.)",
            f"MC - {subdetector}",
        ]

    logger.info("Plotting variable: {}".format(name))

    fig, (up, down) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": (2, 1)},
        sharex=True,
    )
    data_hist, data_bins = np.histogram(data, bins=bins, range=range)
    data_hist_norm, _ = np.histogram(
        data, bins=bins, range=range, density=True
    )
    data_centers = (data_bins[1:] + data_bins[:-1]) / 2
    data_err = np.sqrt(data_hist)
    data_err_norm = data_err / (np.diff(data_bins) * len(data))
    up.errorbar(
        data_centers,
        data_hist_norm,
        yerr=data_err_norm,
        label=labels[0],
        color="k",
        marker="o",
        linestyle="",
        markersize=4,
    )
    up, down = print_mc_hist(
        up,
        down,
        bins,
        range,
        labels[1],
        data_hist,
        data_hist_norm,
        data_centers,
        data_err,
        mc_uncorr,
        "r",
        weights,
    )

    if mc_corr is not None:
        up, down = print_mc_hist(
            up,
            down,
            bins,
            range,
            labels[2],
            data_hist,
            data_hist_norm,
            data_centers,
            data_err,
            mc_corr,
            "b",
            weights,
        )

    if name in ["probe_pfChargedIsoPFPV", "probe_pfPhoIso03"]:
        up.set_yscale("log")
    if name == "probe_sieip" and "transformed" not in extra_name:
        ticks = [-0.0002, -0.0001, 0, 0.0001, 0.0002]
        down.set_xticks(ticks)
        down.set_xticklabels(ticks)
    down.set_xlabel(x_label)
    up.set_ylabel("Normalized yield")
    down.set_ylabel("Ratio")
    down.set_xlim(range[0], range[1])
    down.set_ylim(0.8, 1.2)
    down.axhline(
        1,
        color="grey",
        linestyle="--",
    )
    y_minor_ticks = np.arange(0.8, 1.2, 0.1)
    down.set_yticks(y_minor_ticks, minor=True)
    down.grid(True, alpha=0.4, which="minor")
    up.legend()
    # if probe_pt log scale
    hep.cms.label(
        loc=0, data=True, llabel="Work in Progress", rlabel="", ax=up, pad=0.05
    )
    fig_name = name + "_" + subdetector + extra_name
    if writer_epoch is not None:
        writer, epoch = writer_epoch
        writer.add_figure(fig_name, fig, epoch)
    if cometlogger_epoch is not None:
        comet_logger, epoch = cometlogger_epoch
        comet_logger.log_figure(fig_name, fig, step=epoch)
    if writer_epoch is None and cometlogger_epoch is None:
        for dr in output_dir:
            for ext in ["pdf", "png"]:
                fig.savefig(dr + "/" + fig_name + "." + ext, bbox_inches="tight")
    plt.close(fig)


def sample_and_plot_base(
    test_loader,
    model,
    model_name,
    epoch,
    writer,
    comet_logger,
    context_variables,
    target_variables,
    device,
    pipeline,
    calo,
):
    target_size = len(target_variables)
    with torch.no_grad():
        gen, reco, samples = [], [], []
        for context, target, weights, extra in test_loader:
            context = context.to(device)
            target = target.to(device)
            if "zuko" in model_name:
                sample = model(context).sample()
            else:
                sample = model.sample(num_samples=1, context=context)
            context = context.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            sample = sample.detach().cpu().numpy()
            sample = sample.reshape(-1, target_size)
            gen.append(context)
            reco.append(target)
            samples.append(sample)
    gen = np.concatenate(gen, axis=0)
    reco = np.concatenate(reco, axis=0)
    samples = np.concatenate(samples, axis=0)
    gen = pd.DataFrame(gen, columns=context_variables)
    reco = pd.DataFrame(reco, columns=target_variables)
    samples = pd.DataFrame(samples, columns=target_variables)

    # plot the reco and sampled distributions
    for var in target_variables:
        if device == 0 or type(device) != int:
            dump_main_plot(
                reco[var],
                samples[var],
                variable_conf={
                    "name": var,
                    "title": var,
                    "x_label": var,
                    "bins": 100,
                    "range": transformed_ranges[pipeline][var],
                },
                output_dir="",
                subdetector=calo,
                extra_name=f"_reco_sampled_transformed",
                writer_epoch=(writer, epoch) if writer is not None else None,
                cometlogger_epoch=(comet_logger, epoch) if comet_logger is not None else None,
                labels=["Original", "Sampled"],
            )

    # plot after preprocessing back
    preprocess_dct = test_loader.dataset.pipelines
    reco_back = {}
    samples_back = {}
    with open(f"{script_dir}/../preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        vars_config = {d["name"]: d for d in vars_config}
    for var in target_variables:
        reco_back[var] = (
            preprocess_dct[var]
            .inverse_transform(reco[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        samples_back[var] = (
            preprocess_dct[var]
            .inverse_transform(samples[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    reco_back = pd.DataFrame(reco_back)
    samples_back = pd.DataFrame(samples_back)
    for var in target_variables:
        dump_main_plot(
            reco_back[var],
            samples_back[var],
            variable_conf=vars_config[var],
            output_dir="",
            subdetector=calo,
            extra_name=f"_reco_sampled",
            writer_epoch=(writer, epoch) if writer is not None else None,
            cometlogger_epoch=(comet_logger, epoch) if comet_logger is not None else None,
            labels=["Original", "Sampled"],
        )


def transform_and_plot_top(
    mc_loader,
    data_loader,
    model,
    epoch,
    context_variables,
    target_variables,
    device,
    pipeline,
    calo,
    writer=None,
    comet_logger=None,
    output_dir="",
):
    fff = True
    try:
        model_mc, model_data = model
        fff = False
    except:
        pass
    logger.info("Plotting with fff: {}".format(fff))

    with torch.no_grad():
        data_lst, mc_lst, mc_corr_lst = [], [], []
        data_context_lst, mc_context_lst, mc_corr_context_lst = [], [], []
        mc_weights_lst = []
        data_extra_lst, mc_extr_lst = [], []
        for data, mc in zip(data_loader, mc_loader):
            context_data, target_data, weights_data, extra_data = data
            context_mc, target_mc, weights_mc, extra_mc = mc
            if fff:
                target_mc_corr, _ = model.transform(
                    target_mc, context_mc, inverse=False
                )
            else:  # two flows case
                # zuko
                latent_space_mc = model_mc(context_mc).transform(target_mc)
                target_mc_corr = model_data(context_mc).transform.inv(latent_space_mc)
            target_data = target_data.detach().cpu().numpy()
            target_mc = target_mc.detach().cpu().numpy()
            target_mc_corr = target_mc_corr.detach().cpu().numpy()
            context_data = context_data.detach().cpu().numpy()
            context_mc = context_mc.detach().cpu().numpy()
            weights_mc = weights_mc.detach().cpu().numpy()
            extra_data = extra_data.detach().cpu().numpy()
            extra_mc = extra_mc.detach().cpu().numpy()
            data_lst.append(target_data)
            mc_lst.append(target_mc)
            mc_corr_lst.append(target_mc_corr)
            data_context_lst.append(context_data)
            mc_context_lst.append(context_mc)
            mc_corr_context_lst.append(context_mc)
            mc_weights_lst.append(weights_mc)
            data_extra_lst.append(extra_data)
            mc_extr_lst.append(extra_mc)
    data = np.concatenate(data_lst, axis=0)
    mc = np.concatenate(mc_lst, axis=0)
    mc_corr = np.concatenate(mc_corr_lst, axis=0)
    data = pd.DataFrame(data, columns=target_variables)
    mc = pd.DataFrame(mc, columns=target_variables)
    mc_corr = pd.DataFrame(mc_corr, columns=target_variables)
    data_context = np.concatenate(data_context_lst, axis=0)
    mc_context = np.concatenate(mc_context_lst, axis=0)
    mc_corr_context = np.concatenate(mc_corr_context_lst, axis=0)
    data_context = pd.DataFrame(data_context, columns=context_variables)
    mc_context = pd.DataFrame(mc_context, columns=context_variables)
    mc_corr_context = pd.DataFrame(mc_corr_context, columns=context_variables)
    weights_mc = np.concatenate(mc_weights_lst, axis=0)
    data_extra = np.concatenate(data_extra_lst, axis=0)
    data_extra = pd.DataFrame(data_extra, columns=["probe_energyRaw"])
    mc_extra = np.concatenate(mc_extr_lst, axis=0)
    mc_extra = pd.DataFrame(mc_extra, columns=["probe_energyRaw"])

    # plot the reco and sampled distributions
    for var in target_variables:
        if device == 0 or type(device) != int:
            dump_main_plot(
                data[var],
                mc[var],
                variable_conf={
                    "name": var,
                    "title": var,
                    "x_label": var,
                    "bins": 100,
                    "range": transformed_ranges[pipeline][var],
                },
                output_dir=output_dir,
                subdetector=calo,
                mc_corr=mc_corr[var],
                weights=weights_mc,
                extra_name=f"_top_transformed",
                writer_epoch=(writer, epoch) if writer is not None else None,
                cometlogger_epoch=(comet_logger, epoch)
                if comet_logger is not None
                else None,
                labels=None,
            )

    # sample back
    # note that pipelines are actually the same, trained on data
    data_pipeline = data_loader.dataset.pipelines
    mc_pipeline = mc_loader.dataset.pipelines

    with open(f"{script_dir}/../preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        vars_config = {d["name"]: d for d in vars_config}

    for var in target_variables:
        data[var] = (
            data_pipeline[var]
            .inverse_transform(data[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc[var] = (
            mc_pipeline[var]
            .inverse_transform(mc[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_corr[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_corr[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    for var in context_variables:
        data_context[var] = (
            data_pipeline[var]
            .inverse_transform(data_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_context[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_corr_context[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_corr_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    # photon ID
    # make dataframes by merging context, target and extra
    data_df = pd.concat([data, data_context, data_extra], axis=1)
    mc_df = pd.concat([mc, mc_context, mc_extra], axis=1)
    mc_corr_df = pd.concat([mc_corr, mc_corr_context, mc_extra], axis=1)

    data_df["probe_mvaID"] = calculate_photonid_mva(data_df, calo=calo)
    mc_df["probe_mvaID"] = calculate_photonid_mva(mc_df, calo=calo)
    mc_corr_df["probe_mvaID"] = calculate_photonid_mva(mc_corr_df, calo=calo)

    for var in target_variables + ["probe_mvaID"]:
        if device == 0 or type(device) != int:
            dump_main_plot(
                data_df[var],
                mc_df[var],
                variable_conf=vars_config[var],
                output_dir=output_dir,
                subdetector=calo,
                mc_corr=mc_corr_df[var],
                weights=weights_mc,
                extra_name="_top",
                writer_epoch=(writer, epoch) if writer is not None else None,
                cometlogger_epoch=(comet_logger, epoch)
                if comet_logger is not None
                else None,
                labels=None,
            )

    # now plot profiles
    if device == 0 or type(device) != int:
        nbins = 8
        for column in target_variables + ["probe_mvaID"]:
            for cond_column in context_variables:
                logger.info(
                    "Plotting profile for {} vs {}".format(column, cond_column)
                )
                dump_full_profile_plot(
                    nbins,
                    column,
                    cond_column,
                    data_df,
                    mc_df,
                    mc_corr_df,
                    subdetector=calo,
                    weights=weights_mc,
                    output_dir=output_dir,
                    extra_name="_top",
                    writer_epoch=(writer, epoch) if writer is not None else None,
                    cometlogger_epoch=(comet_logger, epoch) if comet_logger is not None else None,
                )


def plot_one(
    mc_test_loader,
    data_test_loader,
    model,
    epoch,
    writer,
    comet_logger,
    context_variables,
    target_variables,
    device,
    pipeline,
    calo,
):
    with torch.no_grad():
        data_lst, mc_lst, mc_corr_lst = [], [], []
        data_context_lst, mc_context_lst, mc_corr_context_lst = [], [], []
        mc_weights_lst = []
        data_extra_lst, mc_extr_lst = [], []
        for (data_context, data_target, data_weights, data_extra), (
            mc_context,
            mc_target,
            mc_weights,
            mc_extra,
        ) in zip(data_test_loader, mc_test_loader):
            data_context = data_context.to(device)
            data_target = data_target.to(device)
            mc_context = mc_context.to(device)
            mc_target = mc_target.to(device)
            latent_mc = model(mc_context).transform(mc_target)
            # replace the last column in mc_context with 0 instead of 1
            mc_context[:, -1] = 0
            mc_target_corr = model(mc_context).transform.inv(latent_mc)
            data_target = data_target.detach().cpu().numpy()
            data_context = data_context.detach().cpu().numpy()
            data_extra = data_extra.detach().cpu().numpy()
            mc_target = mc_target.detach().cpu().numpy()
            mc_target_corr = mc_target_corr.detach().cpu().numpy()
            mc_context = mc_context.detach().cpu().numpy()
            mc_extra = mc_extra.detach().cpu().numpy()
            mc_weights = mc_weights.detach().cpu().numpy()
            data_lst.append(data_target)
            data_context_lst.append(data_context)
            data_extra_lst.append(data_extra)
            mc_lst.append(mc_target)
            mc_corr_lst.append(mc_target_corr)
            mc_context_lst.append(mc_context)
            mc_weights_lst.append(mc_weights)
            mc_extr_lst.append(mc_extra)
    data = np.concatenate(data_lst, axis=0)
    mc = np.concatenate(mc_lst, axis=0)
    mc_corr = np.concatenate(mc_corr_lst, axis=0)
    data = pd.DataFrame(data, columns=target_variables)
    mc = pd.DataFrame(mc, columns=target_variables)
    mc_corr = pd.DataFrame(mc_corr, columns=target_variables)
    data_context = np.concatenate(data_context_lst, axis=0)
    mc_context = np.concatenate(mc_context_lst, axis=0)
    # remove the last column from mc_context
    mc_context = mc_context[:, :-1]
    data_context = pd.DataFrame(data_context, columns=context_variables)
    mc_context = pd.DataFrame(mc_context, columns=context_variables)
    mc_weights = np.concatenate(mc_weights_lst, axis=0)
    data_extra = np.concatenate(data_extra_lst, axis=0)
    data_extra = pd.DataFrame(data_extra, columns=["probe_energyRaw"])
    mc_extra = np.concatenate(mc_extr_lst, axis=0)
    mc_extra = pd.DataFrame(mc_extra, columns=["probe_energyRaw"])

    for var in target_variables:
        if device == 0 or type(device) != int:
            dump_main_plot(
                data[var],
                mc[var],
                variable_conf={
                    "name": var,
                    "title": var,
                    "x_label": var,
                    "bins": 100,
                    "range": transformed_ranges[pipeline][var],
                },
                output_dir="",
                subdetector=calo,
                mc_corr=mc_corr[var],
                weights=mc_weights,
                extra_name="_one_transformed",
                writer_epoch=(writer, epoch) if writer is not None else None,
                cometlogger_epoch=(comet_logger, epoch) if comet_logger is not None else None,
                labels=None,
            )

    # sample back
    pipeline = mc_test_loader.dataset.pipelines

    with open(f"{script_dir}/../preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        vars_config = {d["name"]: d for d in vars_config}

    for var in target_variables:
        data[var] = (
            pipeline[var].inverse_transform(data[var].values.reshape(-1, 1)).reshape(-1)
        )
        mc[var] = (
            pipeline[var].inverse_transform(mc[var].values.reshape(-1, 1)).reshape(-1)
        )
        mc_corr[var] = (
            pipeline[var]
            .inverse_transform(mc_corr[var].values.reshape(-1, 1))
            .reshape(-1)
        )

    for var in context_variables:
        data_context[var] = (
            pipeline[var]
            .inverse_transform(data_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_context[var] = (
            pipeline[var]
            .inverse_transform(mc_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )

    # photon ID
    # make dataframes by merging context, target and extra
    data_df = pd.concat([data, data_context, data_extra], axis=1)
    mc_df = pd.concat([mc, mc_context, mc_extra], axis=1)
    mc_corr_df = pd.concat([mc_corr, mc_context, mc_extra], axis=1)

    data_df["probe_mvaID"] = calculate_photonid_mva(data_df, calo=calo)
    mc_df["probe_mvaID"] = calculate_photonid_mva(mc_df, calo=calo)
    mc_corr_df["probe_mvaID"] = calculate_photonid_mva(mc_corr_df, calo=calo)

    for var in target_variables + ["probe_mvaID"]:
        if device == 0 or type(device) != int:
            dump_main_plot(
                data_df[var],
                mc_df[var],
                variable_conf=vars_config[var],
                output_dir="",
                subdetector=calo,
                mc_corr=mc_corr_df[var],
                weights=mc_weights,
                extra_name="_one",
                writer_epoch=(writer, epoch) if writer is not None else None,
                cometlogger_epoch=(comet_logger, epoch) if comet_logger is not None else None,
                labels=None,
            )

    # now plot profiles
    if device == 0 or type(device) != int:
        nbins = 8
        for column in target_variables + ["probe_mvaID"]:
            for cond_column in context_variables:
                dump_full_profile_plot(
                    nbins,
                    column,
                    cond_column,
                    data_df,
                    mc_df,
                    mc_corr_df,
                    subdetector=calo,
                    weights=mc_weights,
                    output_dir="",
                    extra_name="_one",
                    writer_epoch=(writer, epoch) if writer is not None else None,
                    cometlogger_epoch=(comet_logger, epoch) if comet_logger is not None else None,
                ) 


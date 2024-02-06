import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import logging
from utils.classifier import classify
import torch
from utils.log import setup_comet_logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@hydra.main(version_base=None, config_path="config_classifier", config_name="test_config")
def main(cfg):
    # /work/tfluehma/git
    initial_dir = get_original_cwd()
    logger.debug("Initial dir: {}".format(initial_dir))
    logger.debug("Current dir: {}".format(os.getcwd()))
    logger.info("Training wit cfg: \n{}".format(OmegaConf.to_yaml(cfg)))

    # Setup Comet logger
    if cfg.logger:
        comet_name = os.getcwd().split("/")[-1]
        comet_logger = setup_comet_logger(comet_name, cfg, project_name_="comet-optimizer")


    # save the config -> run_classifier
    cfg_name = HydraConfig.get().job.name
    with open(f"{os.getcwd()}/{cfg_name}.yaml", "w") as file:
        OmegaConf.save(config=cfg, f=file)

    # Get device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.debug("Using {}".format(device))

    # Optimizer stuff -> later to be put in config
    from comet_ml import Optimizer
    config_opt = {
        "algorithm": "bayes",
        "parameters": {
            "learning_rate": {"type": "float", "scaling_type": "loguniform", "min": 1e-6, "max": 1e-1}, 
            "batch_size": {"type": "discrete", "values": [16, 32, 64, 128, 256]},
        },
        "spec": {
            "maxCombo": 0,
            "metric": "loss",
            "objective": "minimize",
        },
        "name": "Bayesian Search",
        "trials": 1,
    }
    opt = Optimizer(config_opt)
    for experiment in opt.get_experiments():
        experiment.log_parameter("learning_rate", 10)
    # classify(device, cfg, opt)


if __name__ == "__main__":
    main()
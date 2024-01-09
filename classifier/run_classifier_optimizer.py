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
    from comet_ml import Optimizer, Experiment
    
    config_opt = {
        "algorithm": "bayes",
        "parameters": {
            "learning_rate": {"type": "integer", "min": 1, "max": 3},
            "batch_size": {"type": "integer", "min": 1, "max": 3}    
        }
    }
    def objective(params):
        return 2

    # config_opt = {
    #     "algorithm": "bayes",
    #     "parameters": {
    #         "learning_rate": {"type": "float", "scaling_type": "loguniform", "min": 1e-6, "max": 1e-1}, 
    #         "batch_size": {"type": "discrete", "values": [16, 32, 64, 128, 256]},
    #     },
    #     "spec": {
    #         "maxCombo": 0,
    #         "metric": "loss",
    #         "objective": "minimize",
    #     },
    #     "name": "Bayesian Search",
    #     "trials": 1,
    # }
    opt = Optimizer(config_opt, api_key="5OzmIvJNsXYfBCTb5CTYF8Bqy")
    for experiment in opt.get_experiments(project_name="comet-optimizer"):
        hyperparams = experiment.get_parameter("learning_rate")
        score = objective(hyperparams)
        experiment.log_metric("loss", score)
        experiment.end()

        # classify(device, cfg, opt)


if __name__ == "__main__":
    main()
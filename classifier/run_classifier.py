import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import logging
from utils.classifier import classify
import torch
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
    classify(device, cfg)
if __name__ == "__main__":
    main()
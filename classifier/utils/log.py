from comet_ml import Experiment




def setup_comet_logger(name, cfg):
    comet_logger = Experiment(
        api_key="5OzmIvJNsXYfBCTb5CTYF8Bqy",
        workspace="timflu",
        project_name="ssnf-byt-classify",
        #experiment_name="",
        #save_dir="",
    )
    comet_logger.set_name(name)
    for k, v in cfg.items():
        comet_logger.log_parameter(k, v)
    return comet_logger
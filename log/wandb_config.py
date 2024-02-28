import wandb
from pytorch_lightning.loggers import WandbLogger 



wandb_logger = WandbLogger(project='widya_week2', log_model=True)
wandb_logger.experiment.config["key"] = "06ee7ca7307838ddb249c4cda6662d79e7d7d16d"

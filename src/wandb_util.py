import wandb
from src.private_constants import ENTITY, PROJECT


def wandb_init_wrapper(*args, **kwargs):
    return wandb.init(entity=ENTITY, project=PROJECT, *args, **kwargs)

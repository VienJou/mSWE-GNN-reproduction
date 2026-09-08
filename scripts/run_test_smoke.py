"""Run test_model.py's test pipeline with a config path and wandb disabled.
Usage: python plan/run_test_smoke.py plan/config_pretrained_test.yaml"""
import os, sys
sys.path.insert(0, os.getcwd())
import wandb
from lightning.pytorch.loggers import WandbLogger
import test_model as T
from utils.load import read_config
from utils.miscellaneous import fix_dict_in_config

cfg = read_config(sys.argv[1])
logger = WandbLogger(mode="disabled", config=cfg)
_ = logger.experiment  # Lightning>=2 WandbLogger is lazy: force wandb.init() so wandb.config exists
fix_dict_in_config(wandb)
T.main(wandb.config)

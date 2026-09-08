"""Run main.py's full train->validate->test pipeline with an alternative config path
and wandb forced offline. Usage: python plan/run_main_smoke.py plan/config_smoke.yaml"""
import os, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault("WANDB_MODE", "offline")
import wandb
from lightning.pytorch.loggers import WandbLogger
import main as M
from utils.load import read_config
from utils.miscellaneous import fix_dict_in_config

cfg = read_config(sys.argv[1])
M.wandb_logger = WandbLogger(log_model=True, config=cfg)
_ = M.wandb_logger.experiment  # Lightning>=2 WandbLogger is lazy: force wandb.init() so wandb.config exists
fix_dict_in_config(wandb)
M.main(wandb.config)

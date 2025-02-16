import omegaconf
from omegaconf import OmegaConf
import numpy as np
import torch
import random
import transformers
import logging
import sys
import os

def load_config() -> omegaconf.DictConfig:
    args = OmegaConf.from_cli()

    # load default config
    configs = OmegaConf.load(args.configs)
    del args['configs']
    return configs

def setting_seed(seed):
    # setting the seed of CPU
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # setting the seed of GPU
    torch.cuda.manual_seed(seed)

    # setting the seed of transforemers
    transformers.set_seed(seed)

    # set deterministic setting of cuDNN in CUDA     
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logger(dir_path: str = None, file_path: str = None, level: int = logging.INFO):
    """
    Initializes the logging configuration.

    Args:
        dir_path (str, optional): Directory path where the log file will be saved. If None, logs will not be saved to a file.
        file_path (str, optional): The name of the log file. If None, logs will not be saved to a file.
        level (int, optional): Logging level (default: logging.INFO).

    """
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로그 포맷 설정
    log_format = logging.Formatter("INFO: %(message)s")

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    logging.basicConfig(level=level, handlers=[console_handler])

    # 로그 저장
    if dir_path and file_path:
        os.makedirs(dir_path, exist_ok=True)
        log_file_path = os.path.join(dir_path, file_path + ".log")
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(log_format)
        logging.getLogger().addHandler(file_handler)

    logging.info("Logger initialized.")
    return logging.getLogger()
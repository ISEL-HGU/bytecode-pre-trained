# config.py

import json
from typing import NamedTuple, List

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # 'warm up' period = warmup(0.1)*total_steps
    # linearly incresing learning rate from zero to the specified value (5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    adam_epsilon: int = 1e-8
    adam_betas: List[float] = [0.9, 0.999]
    
    

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


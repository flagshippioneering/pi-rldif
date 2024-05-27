import argparse
import yaml
import json
from model.mod_pifold import InverseFoldingDiffusionPiFoldModel
import os
import torch

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(config)

if __name__ == '__main__':
    args = load_config('./configs/config.yaml')
    model = InverseFoldingDiffusionPiFoldModel(args.pifold_model)
    #Check if last.ckpt exists in current directory then wget
    if not os.path.exists('last.ckpt'):
        os.system('wget https://zenodo.org/records/11304952/files/last.ckpt')
    state_dict = torch.load('last.ckpt')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('model.', '')] = v
    model.load_state_dict(new_state_dict)
    import pdb; pdb.set_trace()


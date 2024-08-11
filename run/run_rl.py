#!/usr/bin/env python3
from typing import *
from typing import Any
from lightning_fabric.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from utils.utils import Config, load_config
from typing import *
from model.rl_model import RLStructuralRecoveryModel
from model.rl_diffusion import RLApproachDiffusion
from data.dataset import RLDIFDataset
import pytorch_lightning as pl

# Fix the seed
seed_everything(42)

def main(config):
    dataset = RLDIFDataset(config.data)
    model = RLStructuralRecoveryModel(config.pifold_model).cuda()
    dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=model.collate_fn,
            )
    dataloader = {'train': dataloader, 'val': dataloader}
    # wrap the model in the training strategy
    approach = RLApproachDiffusion(config.train, model, dataloader)

    trainer = pl.Trainer()

    trainer.fit(approach)

if __name__ == "__main__":
    args = load_config('./configs/rl_config.yaml')
    main(args)

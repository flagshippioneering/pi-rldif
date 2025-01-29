from .base import DLCallback, DLTrainer, SingleModelApproach
from utils.common_callbacks import DefaultLogger, PretrainedModelCallback, ValidationCallback, StopOnNaNCallback
import torch
import pytorch_lightning as pl
import numpy as np

def train(config, model, dataloaders):
    sma = SingleModelApproach(config.train, model, dataloaders)

    train_callbacks = [
        DefaultLogger(),
        PretrainedModelCallback(config.train),
        ValidationCallback(
            {k: dl for k, dl in dataloaders.items() if k == "test"}, config.train
        ),
        SamplingCallback(config, dataloaders["test"], name="test"),
        # SamplingCallback(config, dataloaders["single_chain"], name="single_chain"),
        # SamplingCallback(config,  dataloaders["short"], name="short"),
        StopOnNaNCallback(),
    ]

    trainer = DLTrainer(
        config=config,
        approach=sma,
        callbacks=train_callbacks,
        name = config.name
    )
    trainer.train()


class SamplingCallback(DLCallback):
    def __init__(self, config, dataloader, name="test"):
        self.config = config
        self.batch = next(iter(dataloader))
        self.name = name

    @staticmethod
    def get_seq(f: torch.FloatTensor) -> str:
        idcs = torch.argmax(f, dim=-1)
        seq = "".join([IndexedAminoAcids.alias(i) for i in idcs])
        return seq

    def on_train_epoch_end(
        self, trainer: pl.Trainer, approach: SingleModelApproach
    ) -> None:
        # if self.config.data.test and approach.current_epoch > 0:
        #     return

        # if approach.current_epoch % 10:
        #     return
        out = approach.model.sample(self.batch.clone(), closure=True)

        accs = []
        for ft, fp, mask in zip(
            out["features_true"], out["features_0_step"], out["mask"]
        ):
            mask = mask.astype(bool)
            ft, fp = torch.tensor(ft[mask]), torch.tensor(fp[mask])
            n = ft.shape[0]
            acc = (ft.argmax(axis=-1) == fp.argmax(axis=-1)).sum() / float(n)
            accs.append(acc)

        acc = np.mean(accs)

        approach.log(name=f"neg_seq_recovery_acc_{self.name}", value=-acc, batch_size=1)
        print(f"{self.name} Accuracy: {acc}")
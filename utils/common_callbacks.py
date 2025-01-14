from typing import *
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder as _BSF
from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor
from torch import Tensor, is_tensor
from torch.utils.data import DataLoader, DistributedSampler

from ai.dl.models import BaseModel, ComputeMode
from ai.tools.dl import all_reduce_avg_, barrier, find_unused_parameters, t2n

from run.base import BaseApproach, DLCallback, get_last_lr, DLTrainer
from run.base import PatchedModelCheckpoint as ModelCheckpoint
import rrg
import plotly.graph_objects as go

__all__ = [
    "ModelCheckpoint",
    "DLCallback",
    "DefaultLogger",
    "ValidationCallback", 
    "StopOnNaNCallback",
    "BatchSizeFinder",
    "DeviceStatsMonitor",
    "FindUnusedParameters",
    "TrainingInfoCallback",
]

class TrainingInfoCallback(DLCallback):
    def __init__(self, config, max_scatter_points: int = 1000):
        # config should be the top-level config, containing config.model and config.train
        try:
            self.model_name = config.model.name
        except AttributeError:
            raise ValueError("TrainingInfoCallback requires config.model.name be set")

        self.config = config
        self._max_scatter_points = max_scatter_points

    def on_train_start(self, trainer: "DLTrainer", approach: "BaseApproach"):
        report = rrg.Report(f"Training info for {self.model_name}")
        self._add_lr_info(report, trainer, approach)
        #self.s2.store_report(report, self.model_name + "/training_info")

    def _add_lr_info(self, report: "rrg.Report", trainer: "DLTrainer", approach: "BaseApproach"):
        train_config = self.config.train
        total_steps = train_config.n_train_steps
        if total_steps is None:
            if train_config.epoch_train_steps is not None:
                total_steps = train_config.epoch_train_steps * train_config.n_epochs
            else:
                dl = approach.train_dataloader()
                total_steps = len(dl) * train_config.n_epochs
        total_steps //= train_config.opt.gradient_accumulation_steps

        # total_steps = min(total_steps, 50000)

        scheds = approach.lr_schedulers()
        if not isinstance(scheds, list):
            scheds = [scheds]

        lr_plots = {}
        for i_sched, sched in enumerate(scheds):
            sched = deepcopy(sched)
            lr_values = []
            for _ in range(total_steps):
                lr_values.append(sched.get_last_lr()[0])
                sched.step()

            y = np.array(lr_values)
            x = np.arange(len(y))

            if len(x) > self._max_scatter_points:
                # downsample
                idxs = np.linspace(0, len(x) - 1, self._max_scatter_points, dtype=int)
                x = x[idxs]
                y = y[idxs]

            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=x, y=y, mode="lines"))
            fig.update_layout(
                title=f"Schedule {i_sched}",
                xaxis_title="Step",
                yaxis_title="Learning rate",
            )

            lr_plots[f"Schedule {i_sched}"] = fig

        report.add_elements(rrg.SectionHeader("Learning rates"), rrg.Cols(lr_plots))


class BatchSizeFinder(_BSF):
    class Mode():
        power = "power"
        binsearch = "binsearch"

    def __init__(
        self,
        dataloaders: Union[DataLoader, Dict[str, DataLoader], List[DataLoader]],
        train_config,
        discount_factor: float = 0.8,
        mode: Mode = Mode.binsearch,
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
        round_to_nearest: int = 16,
    ) -> None:
        super().__init__(mode, steps_per_trial, init_val, max_trials, batch_arg_name)

        self._dataloaders = dataloaders
        self._discount_factor = discount_factor
        self._round_to_nearest = round_to_nearest
        self.train_config = train_config

    def scale_batch_size(self, trainer: "pl.Trainer", approach: BaseApproach) -> None:
        # set the starting point to the specified value
        self._init_val = approach.config.data.batch_size_per_gpu

        out = super().scale_batch_size(trainer, approach)

        # For some reason, it tends to overestimate. We also want things to be a multiple of 16.
        # So, make some corrections here.
        self.optimal_batch_size = int(self._discount_factor * self.optimal_batch_size)
        if self.optimal_batch_size > self._round_to_nearest * 4:
            # rounding down makes a big diff below 4x, so set a cutoff
            self.optimal_batch_size = self.optimal_batch_size - (self.optimal_batch_size % self._round_to_nearest)
        approach.batch_size = self.optimal_batch_size  # sync it back

        # The inherited method will set the batch size for the train dataloader, but here we have the chance
        # to apply the same to any other dataloaders
        if isinstance(self._dataloaders, dict):
            dls = self._dataloaders.values()
        elif isinstance(self._dataloaders, list):
            dls = self._dataloaders
        else:
            dls = [self._dataloaders]

        for dl in dls:
            # The approach has a function to do this, so call it. Technically, this could be a
            # staticmethod (except for the fact that it has a default batch size, set by the config).
            approach.set_dataloader_batch_size(dl, self.optimal_batch_size)

        logger.info(f"Batch size has been set to {self.optimal_batch_size}")
        self.train_config.data.batch_size_per_gpu = (
            self.optimal_batch_size
        )  # make it easier for other things that reference this to get the right value

        return out


class FindUnusedParameters(DLCallback):
    def __init__(self, every_epoch: bool = True, every_batch: bool = False):
        self.original_state_dict = None
        self.every_epoch = every_epoch
        self.every_batch = every_batch

    def set_state(self, model):
        self.original_state_dict = deepcopy(model.state_dict())

    def on_train_start(self, trainer: "DLTrainer", approach: BaseApproach) -> None:
        self.set_state(approach.model)

    def on_train_epoch_end(self, trainer: "DLTrainer", approach: BaseApproach) -> None:
        if self.every_epoch:
            self.run(approach.model)
            self.set_state(approach.model)

    def on_train_batch_end(
        self,
        trainer: "DLTrainer",
        approach: BaseApproach,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.every_batch:
            self.run(approach.model)

    def on_train_end(self, trainer: "DLTrainer", approach: BaseApproach) -> None:
        self.run(approach.model)

    def run(self, model):
        find_unused_parameters(self.original_state_dict, model.state_dict())


class StopOnNaNCallback(DLCallback):
    def __init__(
        self,
        save_file: str = "/tmp/badstep.ckpt",
        try_inference: bool = False,
        warmup_steps: int = 0,
    ):
        """A callback to stop and inspect training if a NaN loss is encountered.

        If using with AMP, you may want to set `warmup_steps>0`. The first few steps may produce inf losses
        as the loss scaling is being set.

        Some useful information is dumped to `save_file`:
        - the model state dict
        - the loss (so you can tell if it's NaN vs inf vs...)
        - the batch
        If `try_inference` is set:
        - the output of `model.inference_step(batch)`
        - the exception if the above does not work

        Args:
            save_file (str): Path to save the dump file. Defaults to "/tmp/badstep.ckpt".
            try_inference (bool): Whether to attempt an inference step. Defaults to False.
            warmup_steps (int): This callback will only come into effect after `warmup_steps` steps. Defaults to 0.
        """
        self.save_file = save_file
        self.try_inference = try_inference
        self.warmup_steps = warmup_steps
        self.prev_batch = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"]
        if batch_idx >= self.warmup_steps and not (loss == loss):
            model = approach.model
            assert not isinstance(model, Tensor)
            ckpt = {
                "state": model.state_dict(),
                "loss": loss,
            }

            if self.try_inference:
                try:
                    assert not isinstance(model.inference_step, Tensor)
                    ckpt["inference"] = model.inference_step(batch)
                except Exception as e:
                    ckpt["inference_exception"] = e

            ckpt["batch"] = batch

            if self.prev_batch is not None:
                ckpt["prev_batch"] = self.prev_batch

            torch.save(ckpt, self.save_file)
            msg = f"Encountered a NaN loss. State saved to {self.save_file} for inspection. Stopping training."
            logger.error(msg)
            trainer.should_stop = True

        else:
            self.prev_batch = batch

class ValidationCallback(DLCallback):
    def __init__(
        self,
        dataloaders: Union[
            DataLoader[Any],
            Dict[str, DataLoader[Any]],
            Callable[[str], DataLoader[Any]],
        ],
        config,
        dataloader_args: Optional[List[str]] = None,
    ):
        if isinstance(dataloaders, Dict):
            self.val_dls = dataloaders
        elif callable(dataloaders):
            assert dataloader_args, "Must pass dataloader_args if dataloaders is a function."
            self.val_dl_func = dataloaders
            self.val_dl_names = dataloader_args
        else:
            # we want to support any type of dataloader - so we just assume that if its not a dict or a function, it's a single dataloader.
            self.val_dls = {"test": dataloaders}

        self.config = config

    def on_train_epoch_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        if approach.models is None or "validation_model" not in approach.models:
            return
        model = approach.models["validation_model"]
        if not hasattr(self, "val_dls"):
            self.val_dls = {x: self.val_dl_func(x, approach) for x in self.val_dl_names}
        val_losses: Dict[str, Any] = {k: 0.0 for k in self.val_dls}

        for dl in self.val_dls.values():
            if hasattr(dl, "sampler") and isinstance(dl.sampler, DistributedSampler):
                dl.sampler.set_epoch(trainer.current_epoch)

        if self.config.epoch_val_steps is None:
            epoch_val_steps = {k: np.inf for k in self.val_dls}
        elif isinstance(self.config.epoch_val_steps, int):
            epoch_val_steps = {k: self.config.epoch_val_steps for k in self.val_dls}
        else:
            raise ValueError("Passing a dictionary to config.epoch_val_steps not supported")

        model.eval()
        for k_val, val_dl in self.val_dls.items():
            n_batch = 0
            if epoch_val_steps[k_val] < np.inf:
                total = epoch_val_steps[k_val]
            else:
                try:
                    total = len(val_dl)
                except TypeError:
                    total = None

            for n_batch, batch in enumerate(
                model.lqdm(
                    val_dl,
                    desc=f"Epoch {trainer.current_epoch}: {k_val.upper()}",
                    total=total,
                    disable=self.config.verbosity < 10,
                )
            ):
                trainer.strategy.barrier()
                val_output = model.validation_step(batch)
                if isinstance(val_output, Tensor):
                    val_losses[k_val] += val_output.item()
                else:
                    val_losses[k_val] += val_output["loss"].item()

                if n_batch >= epoch_val_steps[k_val]:
                    break
            loss = val_losses[k_val] / (n_batch + 1)
            if trainer.world_size > 1:
                trainer.strategy.barrier()
                if not isinstance(loss, float):
                    loss = all_reduce_avg_(loss)
                if isinstance(loss, Tensor):
                    loss = loss.item()
            val_losses[k_val] = loss
        for key, loss in val_losses.items():
            assert not isinstance(
                loss, List
            ), f"Only logging of single values is supported but {key} contains a list of values. If this is returned when using INFERENCE, you need to reduce the list to a single value before logging it."
            approach.log(name=f"epoch_{key}", value=loss, batch_size=1, sync_dist=True)

        if self.config.verbosity > 0:
            logger.info(f"Epoch {approach.current_epoch} validation losses:")
            for k, v in val_losses.items():
                logger.info(f"   {k} = {v}")

class DefaultLogger(DLCallback):
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.is_global_zero:
            approach.log("train_loss_batch", outputs["loss"])
            # approach.log("epoch_train", outputs["loss"], on_epoch=True, batch_size=approach.config.data.batch_size_per_gpu)
            schedulers = approach.lr_schedulers()
            if schedulers is not None:
                if not isinstance(schedulers, List):
                    lr = get_last_lr(schedulers, approach.config)
                    # lr = schedulers.get_last_lr()[0]  # type: ignore
                    approach.log(f"learning_rate", lr)
                else:
                    for idx, s in enumerate(schedulers):
                        lr = get_last_lr(s, approach.config)
                        # lr = s.get_last_lr()[0]  # type: ignore
                        approach.log(f"learning_rate_{idx}", lr)

    def on_train_epoch_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        if approach.avg_epoch_loss is not None:
            approach.log(f"epoch_train", approach.avg_epoch_loss, batch_size=1, sync_dist=True)

        # if approach.models is not None:
        #     for model_name in approach.models.keys():
        #         model = approach.models[model_name]
        #         approach.log_state(model)


class PerplexityCallback(ValidationCallback):
    # inherits from ValidationCallback (instead of DLCallback standard) because we need `epoch_test` values and want to prevent order sensitivity in training scripts
    def on_train_epoch_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        super().on_train_epoch_end(trainer, approach)
        metrics = trainer.callback_metrics
        train_loss = metrics.get("epoch_train")
        if isinstance(train_loss, Tensor):
            self.train_perplexity = np.exp(train_loss.item())
            approach.log("perplexity_train", self.train_perplexity, sync_dist=True)
        test_loss = metrics.get("epoch_test")
        if isinstance(test_loss, Tensor):
            self.test_perplexity = np.exp(test_loss.item())
            approach.log("perplexity_test", self.test_perplexity, sync_dist=True)


class GenericTrainOutputLogger(DLCallback):
    def __init__(self, keys: Optional[List[str]] = None) -> None:
        self.keys = keys
        super().__init__()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if hasattr(approach, "model") and trainer.is_global_zero:
            assert isinstance(approach.model, BaseModel)
            # TODO: What is last_output? I can't find a reference to it anywhere - Ian
            # TODO: no idea, probably vestigial - SN
            output = approach.model.last_output
            if self.keys is not None:
                for key in self.keys:
                    approach.log(f"train_{key}", output[key])

class PretrainedModelCallback(DLCallback):
    def __init__(self, config):
        self.config = config
        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        if self.config.pretrained_model.name is not None:
            cfg = self.config.pretrained_model
            approach.model.load_checkpoint(
                cfg.name,
                best=(cfg.best or None),
                last=cfg.last,
                epoch=cfg.epoch,
                step=cfg.step,
                version=cfg.version,
                strict=False,
                size_strict=False,
                verbose=True,
            )
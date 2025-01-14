from glob import glob
from typing import Any, Callable, Dict, List, Optional, Union
from torch.optim import Optimizer
from torch.utils.data import DataLoader,DistributedSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.strategies import single_device
from torch.optim import Adam, AdamW, SGD
from pytorch_lightning.loggers import (
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)
import wandb
import pandas as pd
from torch.profiler.profiler import schedule, tensorboard_trace_handler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loops.training_epoch_loop import _TrainingEpochLoop
import torch.Tensor as Tensor
import os
from loguru import logger
import lightning_fabric as lf

from utils.nemo import EMACallback, PatchedModelCheckpoint
from utils.lr_schedules import lr_schedules

# The base LightningModule that all approaches inherit from. Should not be instantiated on its own.
#: Optional[Dict[str, BaseModel]]
class BaseApproach(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.current_epoch_steps = 0
        self.models = None
        self.avg_epoch_loss: Optional[float] = None
        self.stop_epoch_early = False
        self.verbosity = config.verbosity

    @property
    def batch_size(self) -> int:
        return self.config.data.batch_size_per_gpu

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        if self.stop_epoch_early:
            return -1

    def set_dataloader_batch_size(
        self, dl: DataLoader, batch_size: int = None
    ) -> DataLoader:
        batch_size = batch_size or self.batch_size
        if dl.batch_size == batch_size:
            return dl
        else:
            dl._DataLoader__initialized = False
            dl.batch_size = batch_size
            try:
                dl.batch_sampler.batch_size = batch_size
            except AttributeError:
                pass
            # dl.batch_sampler = BatchSampler(dl.sampler, batch_size, dl.drop_last)
            dl._DataLoader__initialized = True
            return dl

    def set_dataloader_dist_sampler(self, dl: DataLoader) -> DataLoader:
        if (
            not isinstance(self.trainer.strategy, single_device.SingleDeviceStrategy)
            and dl.sampler is RandomSampler
        ):
            dl._DataLoader__initialized = False
            dl.sampler = DistributedSampler(
                dataset=dl.dataset, shuffle=True, drop_last=dl.drop_last
            )
            dl._DataLoader__initialized = True
            return dl
        else:
            return dl

    def train_dataloader(self) -> Any:
        pass

    def configure_optimizers(self) -> Any:
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def get_opt(self, params: Any, config) -> Optimizer:
        opt_type = config.opt.optimizer

        kw = dict()
        if config.lr.lr is not None:
            kw["lr"] = config.lr.lr

        if (opt_type == "adam") or (opt_type is None):
            opt = Adam(params, **kw)
        elif opt_type == "adamW":
            opt = AdamW(params, **kw)
        elif opt_type == "sgd":
            opt = SGD(params, **kw)
        else:
            raise NotImplementedError
        return opt

    def get_lr(
        self,
        optimizer: Optimizer,
        train_dataloader,
        step_scale_factor: int,
        config,
    ) -> Dict[str, Any]:
        sched_type = config.lr.schedule

        d = {"interval": "step", "frequency": 1}

        if sched_type == "constant":
            sched = lr_schedules.get_constant_schedule(optimizer)

        elif sched_type == "poly":
            if config.lr.gamma is None:
                raise ValueError(
                    "No gamma found. If you are using an poly learning rate, you must supply a gamma value in the LRConfig "
                    "(ie. config.train.lr.gamma)."
                )
            sched = lr_schedules.get_polynomial_decay_schedule_with_warmup_dynamic(
                optimizer,
                num_warmup_steps=config.lr.warmup_steps,
                num_training_steps=config.n_train_steps,
                train_dataloader=self.dataloaders["train"],
                step_scale_factor=step_scale_factor,
                power=config.lr.gamma,
                **config.lr.lr_args,
            )

        elif sched_type == "decay_on_plateau":
            # Bit of a hack, since we run our validation callbacks after lightning wants to run
            # scheduler.step. So we tell lightning to check after every batch. Therefore, the
            # patience has to be in terms of batches. Note that this means the LR actually decays
            # after the first step of the subsequent epoch (negligible difference)
            if config.epoch_train_steps is not None:
                spe = config.epoch_train_steps
            else:
                spe = len(self.dataloaders["train"])

            kw = dict(config.lr.lr_args)
            kw.setdefault("verbose", True)
            kw.update(
                factor=config.lr.gamma,
                patience=config.lr.patience * spe,
            )

            for k, v in kw.items():
                print(f"    {k}={v}", flush=True)

            sched = lr_schedules.ReduceLROnPlateau(optimizer, **kw)

            d["monitor"] = config.lr.metric
            d["strict"] = False  # for the first epoch, we won't see anything

        else:
            raise RuntimeError(
                f"Did not know how to construct scheduler for sched_type={sched_type}"
            )

        d["scheduler"] = sched

        return d

class SingleModelApproach(BaseApproach):
    def __init__(
        self,
        config,
        model,
        dataloaders: Optional[Union[Dict[str, DataLoader[Any]], Callable[[str], DataLoader[Any]]]],
    ) -> None:
        super().__init__(config)
        self.models = {"validation_model": model}

        if dataloaders is None:
            print("No dataloaders found - approach configured for inference mode only")
        elif isinstance(dataloaders, Callable):
            self.dataloader_func = dataloaders
        else:
            self.dataloaders = dataloaders

        self.model = model

    def train_dataloader(self) -> DataLoader[Any]:
        if not hasattr(self, "dataloaders"):
            self.dataloaders = {"train": self.dataloader_func("train", self)}
        if self.config.data.reload_dataloaders_every_n_epochs > 0 and self.trainer.current_epoch > 0:
            if hasattr(self, "dataloader_func"):
                self.dataloaders = {"train": self.dataloader_func("train", self)}
            else:
                raise RuntimeError(
                    "No dataloader function found - cannot reload dataloaders. Either provide a dataloader function instead of a dataloader or set config.train.data.reload_every_n_epochs to 0"
                )
        dl = self.set_dataloader_batch_size(self.dataloaders["train"])
        dl = self.set_dataloader_dist_sampler(self.dataloaders["train"])
        return dl

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, Dict[str, Any]]]:
        # breakpoint()
        if not hasattr(self, "dataloaders"):
            self.dataloaders = {"train": self.dataloader_func("train", self)}

        optimizer = self.get_opt(self.model.parameters(), self.config)
        self.model.opt = optimizer

        step_scale_factor = 1
        # okay so sometimes datasets don't have a length...
        if self.config.epoch_train_steps is not None:
            step_scale_factor *= self.config.epoch_train_steps
        if self.config.n_epochs is not None:
            step_scale_factor *= self.config.n_epochs
        step_scale_factor /= self.config.opt.gradient_accumulation_steps
        step_scale_factor /= self.trainer.num_nodes * self.trainer.num_devices

        scheduler = self.get_lr(
            optimizer,
            train_dataloader=self.dataloaders["train"],
            step_scale_factor=step_scale_factor,
            config=self.config,
        )
        self.model.lr_sched = scheduler["scheduler"]

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch: Any):
        # There are 3 functions within a BaseModel where we can jack in to get the loss.
        # 1) is model.step, which is supposed to return the raw loss and nothing else.
        # 2) is model._forward_backward, which additionally scales based on accumulate_grad and applies regularization.
        # 3) is model.training_step, which additionally uses the loss to perform updates using the optimizer.
        # We probably want to drop (2) and (3)
        # Note from SN: Agreed on dropping (2) [it's vestigial at this point], but I'd rather just simplify training_step()
        # to look like validation_step(): just call self.train() and pass the mode to step()
        # loss = self.model.step(batch, mode=ComputeMode.TRAIN)  # type: ignore
        # return loss

        # taking option (3)
        return self.model.training_step(batch)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

def model_checkpointer_args_from_config(
    config, **overrides
) -> dict:
    filefmt = (
        "{epoch:04d}---{step:010d}---"
        + config.checkpointing.mode
        + "---{"
        + config.checkpointing.metric
        + ":.5g}"
    )
    kw = dict(
        filename=filefmt,
        monitor=config.checkpointing.metric,
        mode=config.checkpointing.mode,
        verbose=True,
        save_last=True,
        save_top_k=config.checkpointing.n_best_to_save,
        auto_insert_metric_name=True,
        save_weights_only=False,
        every_n_epochs=config.checkpointing.save_every_n_epochs,
    )
    kw.update(overrides)
    return kw

class DLTrainer:
    def __init__(
        self,
        config,
        approach: BaseApproach,
        callbacks: List[pl.Callback],
        name: Optional[str] = None,
        pl_trainer_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            train_config: configs.TrainConfig = config.train  # type: ignore
        except AttributeError:
            raise ValueError(
                "Config passed to DLTrainer does not have a config.train subconfig."
            )

        try:
            env_config: configs.EnvironmentConfig = config.env  # type: ignore
        except AttributeError:
            raise ValueError(
                "Config passed to DLTrainer does not have a config.env subconfig."
            )

        try:
            self.name = name or config.model.name  # type: ignore
            assert self.name is not None
        except (AttributeError, AssertionError) as e:
            raise ValueError(
                "Model name was not explicitly specified and config.model.name does not exist."
            )

        self.continue_training = train_config.continue_training

        # LIGHTNING TRAINER ARGUMENTS
        lightning_trainer_params = self._get_lightning_params_from_config(
            train_config=train_config,
            env_config=env_config,
            automatic_optimization=approach.automatic_optimization,
        )

        if pl_trainer_args:
            lightning_trainer_params.update(pl_trainer_args)

        # extract the run version so we can pass it to ModelCheckpoint before creating the Trainer

        # CALLBACK STUFF
        # add early stopping
        es_b = train_config.early_stopping.patience
        if es_b is not None and es_b > 0:
            es = EarlyStopping(
                monitor=train_config.early_stopping.metric
                or train_config.checkpointing.metric,
                mode=train_config.early_stopping.mode
                or train_config.checkpointing.mode,
                patience=es_b,
            )
            callbacks.append(es)

        # add checkpointing if not provided
        # if the for loop did not terminate early, we need to add a callback
        bare_ckpt_tag = f"./models/{self.name}"
        ckpt_tag = f"./models/{self.name}/{self.run_version}"

        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                # found it, move on
                mc = cb
                break
        else:
            # below block shouldn't be needed, since we're always using the latest run version
            # if not self.continue_training:
            #     # delete the old directory if we are not continuing training
            #     s2.delete_tag(ckpt_tag)

            mc = PatchedModelCheckpoint(
                # this creates the directory whether or not it is needed. If we simply set the dirpath but did not create the version, we would be fine. So. I wonder...
                dirpath=ckpt_tag,
                **model_checkpointer_args_from_config(train_config),
            )
            callbacks.append(mc)

            # ADDING NEW CALLBACK TO SAVE EVERY N STEPS
            checkpoint_callback = PatchedModelCheckpoint(
                dirpath=ckpt_tag,
                filename="{step}",
                save_top_k=-1,
                every_n_train_steps=1000,
            )
            callbacks.append(checkpoint_callback)

        if train_config.ema:
            # not None and > 0
            # TODO: we get an illegal memory access when cpu_offload=False. Figure out why
            callbacks.insert(0, EMACallback(train_config.ema, cpu_offload=True))

        # The CoreCallback is effectively part of the DLTrainer,
        # always runs, and cannot be modified.

        callbacks.insert(0, CoreCallback())
        
        # lightning_trainer_params["devices"] = [0, 1, 2, 3]
        if "gpus" in lightning_trainer_params:
            lightning_trainer_params["devices"] = lightning_trainer_params["gpus"]
            # lightning_trainer_params["devices"] = ast.literal_eval(config.train.mpnn_gpus)
            del lightning_trainer_params["gpus"]

        self.trainer = pl.Trainer(**lightning_trainer_params, callbacks=callbacks)

        loop_class = get_loop(train_config)
        min_steps = self.trainer.fit_loop.epoch_loop.min_steps
        max_steps = self.trainer.fit_loop.epoch_loop.max_steps
        unconnected_loop = loop_class(
            self.trainer, min_steps=min_steps, max_steps=max_steps
        )
        self.trainer.fit_loop.epoch_loop = unconnected_loop

        self.ckpt_dir = mc.dirpath
        if self.trainer.global_rank == 0:
            #s2.get_path(str(ckpt_tag), mkdir=s2.MkdirOptions.directory)
            config.write(str(self.ckpt_dir) + "/full_config.yaml")
            config.model.write(str(self.ckpt_dir) + "/config.yaml")  # type: ignore

        if self.trainer.global_rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            config.write(self.log_dir + "/config.yaml")

        # approach.set_logger(self.trainer.logger)
        self.approach = approach

        self.train_config = train_config
        self.env_config = env_config
        self.config = config

        try:
            self.node_rank = os.environ["NODE_RANK"]
        except:
            self.node_rank = 0

    def _get_lightning_params_from_config(
        self,
        # s2,
        train_config,
        env_config,
        automatic_optimization: bool,
    ) -> Dict[str, Any]:
        d = {}

        #Commented this out
        # d["enable_progress_bar"] = False

        log_path = train_config.log_path + self.name
        versions = glob(log_path + "/version_*")

        if not versions:
            version = 0
        else:
            version = max([int(v.split("_")[-1]) for v in versions]) + 1
        if self.continue_training:
            assert (
                version > 0
            ), "continue_training is True, but no previous versions found"
            version -= 1

        self.run_version = f"version_{version}"
        self.log_dir = log_path + "/" + self.run_version


        if train_config.logger == "tensorboard":
            logger = TensorBoardLogger(
                self.log_dir, name=self.name, flush_secs=60
            )
            logger._version = version
        elif train_config.logger == "wandb":
            logger = WandbLogger(
                project=self.name, group=f"DDP-{wandb.util.generate_id()}"
            )
            if type(logger.experiment) is not lf.loggers.logger._DummyExperiment:
                logger.experiment.config.update(
                    pd.json_normalize(vars(train_config), sep="_").to_dict(
                        orient="records"
                    )[0]
                )
                logger.experiment.config.update(
                    pd.json_normalize(vars(env_config), sep="_").to_dict(
                        orient="records"
                    )[0]
                )
        elif train_config.logger == "mlflow":
            logger = MLFlowLogger(
                experiment_name=self.name,
                run_name=self.run_version,
                save_dir=f"logs/mlflow",
                log_model=False,
            )
        # d["logger"] = logger

        # if wandb_logger is None:
        d["logger"] = [logger]
        # else:
        #     d["logger"] = [logger, wandb_logger]

        d["log_every_n_steps"] = train_config.tb_log_frequency

        if train_config.n_epochs:
            d["max_epochs"] = train_config.n_epochs

        if train_config.n_train_steps:
            d["max_steps"] = train_config.n_train_steps

        d["enable_checkpointing"] = True

        if automatic_optimization:
            # Trainer only accepts these args if auto opt. Otherwise, the approach needs to handle them.
            if train_config.opt.clip_grad_norm:
                d["gradient_clip_val"] = train_config.opt.clip_grad_norm
                d["gradient_clip_algorithm"] = "norm"

            if train_config.opt.gradient_accumulation_steps:
                d["accumulate_grad_batches"] = (
                    train_config.opt.gradient_accumulation_steps
                )

        d["precision"] = train_config.precision

        if train_config.strategy is not None:
            d["strategy"] = train_config.strategy

        if env_config.n_gpus > 0:
            #d["accelerator"] = "gpu"
            d["devices"] = env_config.n_gpus
            if train_config.strategy is None:
                if (env_config.devices is None and env_config.n_gpus > 1) or (
                    env_config.devices is not None and len(env_config.devices) > 1
                ):
                    d["strategy"] = "ddp"

        d["devices"] = env_config.devices or "auto"

        d["reload_dataloaders_every_n_epochs"] = (
            train_config.data.reload_dataloaders_every_n_epochs
        )

        # if train_config.profiler.enabled:
        #     sched = schedule(
        #         wait=train_config.profiler.wait,
        #         warmup=train_config.profiler.warmup,
        #         active=train_config.profiler.active,
        #         repeat=train_config.profiler.repeat,
        #     )
        #     # fix this - we should be able to avoid log_path here now that the full config is available
        #     if train_config.profiler.log_path is not None:
        #         profiler = PyTorchProfiler(
        #             schedule=sched,
        #             record_shapes=train_config.profiler.record_shapes,
        #             profile_memory=train_config.profiler.profile_memory,
        #             with_stack=train_config.profiler.with_stack,
        #             with_flops=train_config.profiler.with_flops,
        #             on_trace_reader=tensorboard_trace_handler(
        #                 train_config.profiler.log_path
        #             ),
        #         )
        #     else:
        #         profiler = PyTorchProfiler(
        #             schedule=sched,
        #             record_shapes=train_config.profiler.record_shapes,
        #             profile_memory=train_config.profiler.profile_memory,
        #             with_stack=False,  # this is giving a segmentation fault when True...
        #             with_flops=train_config.profiler.with_flops,
        #         )
        #     d["profiler"] = profiler

        return d

    def train(self) -> None:
        logger.info(f"Starting train processes: {self.name}")
        logger.info(f"    Logs: {self.log_dir}")
        logger.info(f"    Checkpoints: {self.ckpt_dir}")

        # with float32_matmul_precision(self.train_config.float32_matmul_precision):
        if self.continue_training:
            logger.info(
                "Attempting to continue training from last known checkpoint"
            )
            path = self._get_latest_checkpoint()
            self.trainer.fit(self.approach, ckpt_path=path)
        else:
            self.trainer.fit(self.approach)

    def _get_latest_checkpoint(self) -> str:
        if self.train_config.ema:
            return self.ckpt_dir + "/last-EMA.ckpt"
        else:
            return self.ckpt_dir + "/last.ckpt"
        
import pytorch_lightning as pl

class DLCallback(pl.Callback):
    def on_fit_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when fit begins."""

    def on_fit_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when fit ends."""

    def on_sanity_check_start(
        self, trainer: pl.Trainer, approach: BaseApproach
    ) -> None:
        """Called when the validation sanity check starts."""

    def on_sanity_check_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the validation sanity check ends."""

    def on_train_batch_start(
        self, trainer: pl.Trainer, approach: BaseApproach, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """

    def on_train_epoch_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, approach: BaseApproach
    ) -> None:
        """Called when the val epoch begins."""

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, approach: BaseApproach
    ) -> None:
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the test epoch begins."""

    def on_test_epoch_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the test epoch ends."""

    def on_predict_epoch_start(
        self, trainer: pl.Trainer, approach: BaseApproach
    ) -> None:
        """Called when the predict epoch begins."""

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, approach: BaseApproach, outputs: List[Any]
    ) -> None:
        """Called when the predict epoch ends."""

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch begins."""

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""

    def on_test_batch_start(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch begins."""

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""

    def on_predict_batch_start(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch begins."""

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends."""

    def on_train_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the train begins."""

    def on_train_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the train ends."""

    def on_validation_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the validation loop begins."""

    def on_validation_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the validation loop ends."""

    def on_test_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the test begins."""

    def on_test_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the test ends."""

    def on_predict_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when the predict begins."""

    def on_predict_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called when predict ends."""

    def on_exception(
        self, trainer: pl.Trainer, approach: BaseApproach, exception: BaseException
    ) -> None:
        """Called when any trainer execution is interrupted by an exception."""

    def on_save_checkpoint(
        self, trainer: pl.Trainer, approach: BaseApproach, checkpoint: Dict[str, Any]
    ) -> None:
        r"""
        Called when saving a checkpoint to give you a chance to store anything else you might want to save.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            approach: the current :class:`~pytorch_lightning.core.module.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.
        """

    def on_load_checkpoint(
        self, trainer: pl.Trainer, approach: BaseApproach, checkpoint: Dict[str, Any]
    ) -> None:
        r"""
        Called when loading a model checkpoint, use to reload state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            approach: the current :class:`~pytorch_lightning.core.module.LightningModule` instance.
            checkpoint: the full checkpoint dictionary that got loaded by the Trainer.
        """

    def on_before_backward(
        self, trainer: pl.Trainer, approach: BaseApproach, loss: Tensor
    ) -> None:
        """Called before ``loss.backward()``."""

    def on_after_backward(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        """Called after ``loss.backward()`` and before optimizers are stepped."""

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        optimizer: Optimizer,
    ) -> None:
        """Called before ``optimizer.step()``."""

    def on_before_zero_grad(
        self, trainer: pl.Trainer, approach: BaseApproach, optimizer: Optimizer
    ) -> None:
        """Called before ``optimizer.zero_grad()``."""


# This is a special callback that DLTrainer will always include. The logic is this: There will be some callback-like
# operations that we always want to run while training. However, we can't just place these in the BaseApproach, since
# those pseudocallback methods can be overwritten by anyone writing a new approach. We can't guarantee that the writer
# will know to call super().
class CoreCallback(DLCallback):
    def __init__(self) -> None:
        self.train_loss = 0
        super().__init__()
        self.old_max_batches = None

    def on_train_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        pass

    def on_train_epoch_start(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        if self.old_max_batches:
            approach.trainer.fit_loop.max_batches = self.old_max_batches
            self.old_max_batches = None

        approach.set_training_pbar(trainer)

        # LOGGING
        # approach.log_model_states()
        # if approach.logger is not None:
        #     approach.logger.finalize("success")
        # / LOGGING

    def on_train_batch_start(
        self, trainer: pl.Trainer, approach: BaseApproach, batch: Any, batch_idx: int
    ) -> None:
        if approach.config.epoch_train_steps is not None:
            if approach.current_epoch_steps >= approach.config.epoch_train_steps:
                approach.stop_epoch_early = True

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.train_loss += outputs["loss"]
        if approach.verbosity > 10:
            logger.info(f"LOSS={outputs['loss']:.5g}")

        if approach.config.epoch_train_steps is not None:
            if approach.current_epoch_steps >= approach.config.epoch_train_steps:
                self.old_max_batches = approach.trainer.fit_loop.max_batches
                approach.trainer.fit_loop.max_batches = batch_idx + 1
                approach.on_train_epoch_end()
                return None

    def on_after_backward(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        if approach.pbar is not None:
            approach.pbar.update(1)

        return super().on_after_backward(trainer, approach)

    def on_train_epoch_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        approach.avg_epoch_loss = self.train_loss / approach.current_epoch_steps
        self.train_loss = 0
        approach.current_epoch_steps = 0

        if approach.config.verbosity > 0:
            logger.info(
                f"Epoch {approach.current_epoch} average train loss = {approach.avg_epoch_loss}"
            )

        if (
            approach.config.stop_after_n_epochs
            and approach.current_epoch + 1 >= approach.config.stop_after_n_epochs
        ):
            logger.info(
                f"Hit stop_after_n_epochs={approach.config.stop_after_n_epochs} epochs, stopping!"
            )
            trainer.should_stop = True

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        approach: BaseApproach,
        optimizer: Optimizer,
    ) -> None:
        # Console log the learning rate
        schedulers = approach.lr_schedulers()
        if schedulers is not None:
            if not isinstance(schedulers, List):
                schedulers = [schedulers]
            for idx, s in enumerate(schedulers):
                lr = get_last_lr(s, approach.config)
                if approach.verbosity > 10:
                    logger.info(f"LR_{idx}={lr:.4g}")

        # TODO - what is a step? based on optimizer or trainer?
        approach.current_epoch_steps += 1

    def on_train_end(self, trainer: pl.Trainer, approach: BaseApproach) -> None:
        # LOGGING
        # approach.log_model_states()
        if approach.logger is not None:
            approach.logger.finalize("success")
        # / LOGGING


def get_last_lr(schedule, train_config) -> float:
    if isinstance(schedule, lr_schedules.ReduceLROnPlateau):
        # this one's complicated
        try:
            return schedule._last_lr[0]
        except AttributeError:
            # might not have called step yet
            return train_config.lr.lr  # peak LR
    else:
        return schedule.get_last_lr()[0]  # type: ignore

def get_loop(config):
    class NewLoop(_TrainingEpochLoop):
        def __init__(
            self,
            trainer: pl.Trainer,
            min_steps: Optional[int] = None,
            max_steps: int = -1,
        ) -> None:
            self.first_time = True
            super().__init__(trainer=trainer, min_steps=min_steps, max_steps=max_steps)

        def on_run_start(self, data_fetcher) -> None:

            # 1 - what happens when it runs out of juice?
            if self.first_time or not config.continuous_dataloader:
                iter(data_fetcher)
                self.first_time = False
            data_fetcher.fetched += self.batch_progress.current.ready
            data_fetcher._start_profiler = self._on_before_fetch
            data_fetcher._stop_profiler = self._on_after_fetch

    return NewLoop
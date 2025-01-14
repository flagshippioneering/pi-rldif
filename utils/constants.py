from typing import *
import torch


class NamedConfig():
    name: str


class Optimizers():
    adam = "adam"
    adamW = "adamW"
    sgd = "sgd"
    dummy = "dummy"


class OptimizerConfig():
    clip_grad_norm: Optional[float] = 5.0
    clip_grad_by_param: bool = False
    gradient_accumulation_steps: int = 1
    optimizer: Optional[Optimizers] = Optimizers.adam


class LRSchedules():
    constant = "constant"
    # constant_warmup = "constant_warmup"
    # linear = "linear"
    # cosine = "cosine"
    # cosine_restarts = "cosine_restarts"
    poly = "poly"
    # exp = "exp"
    decay_on_plateau = "decay_on_plateau"
    # piecewise_linear = "piecewise_linear"


class LRConfig():
    lr: float

    warmup_fraction: Optional[float] = None  # deprecated?
    warmup_steps: int = 0
    gamma: Optional[float] = None  # decay factor for any schedule that decays
    metric: str = "epoch_test"  # decay_on_plateau metric to track
    patience: int = (
        10  # decay_on_plateau only decay if this many epochs fail to improve
    )

    schedule: LRSchedules = LRSchedules.constant
    lr_args: dict = {}


class DataLoaderConfig():
    # default batch size
    batch_size_per_gpu: Optional[int] = None

    # if provided, can be used instead of batch_size_per_gpu for non-training
    # steps
    inference_batch_size_per_gpu: Optional[int] = None

    # DataLoader workers. 0=dataloader is not forked. >0=fork N processes
    n_workers: int = 1

    # DEPRECATED
    # if provided and we are using BlockedDistributedSampler, will be used to
    # define the number of blocks that are cached. if not provided, will default
    # to n_workers.
    n_cached_blocks: Optional[int] = None
    reload_dataloaders_every_n_epochs: int = 0


class ModelProfilingConfig():
    enabled: bool = False
    # options documented in https://pytorch.org/docs/stable/profiler.html
    wait: int = 10
    warmup: int = 10
    active: int = 10
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True
    log_path: Optional[str] = None


class ImprovementMode():
    min = "min"
    max = "max"


class ModelCheckpointingConfig():
    n_best_to_save: int = -1  # set to -1 to save all
    save_every_n_epochs: Optional[int] = None
    metric: str = "epoch_test"
    mode: ImprovementMode = ImprovementMode.min


class PretrainedModelConfig():
    name: Optional[str] = None
    best: bool = True
    last: Optional[bool] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    version: Optional[int] = None


class EarlyStoppingConfig():
    patience: Optional[int] = None  # if not provided, early stopping is off
    metric: Optional[str] = None  # if not provided, use checkpointing.metric
    mode: Optional[ImprovementMode] = None


class Precision():
    fp32 = 32
    fp16 = "16-mixed"
    bf16 = "bf16-mixed"


class Matmul32Precision():
    highest = "highest"
    high = "high"
    medium = "medium"


class LoggerType():
    tensorboard = "tensorboard"
    wandb = "wandb"
    mlflow = "mlflow"


class TrainConfig():
    # The maximum number of epochs to train. Note that if you have any form of early stopping,
    # this limit will likely not be reached.
    n_epochs: Optional[int] = None
    stop_after_n_epochs: Optional[int] = None  # for debugging

    # The maximum number of optimization steps to train. Works similarly to n_epochs.
    n_train_steps: Optional[int] = None
    # stop_after_n_steps: Optional[int] = None  # for debugging <- doesn't work yet

    # The number of optimization steps per epoch. Normally, this is determined by the length
    # of the dataloader - epochs end when the dataloader is exhausted. This setting is useful
    # if using a very long or infinite dataloader to define arbitrary epoch boundary and trigger
    # all the various epoch end callbacks.
    epoch_train_steps: Optional[int] = None

    # As above, but for any validation callbacks. This can either be a single value that applies
    # to alation datasets or a dictionary with values for each validation dataset.
    epoch_val_steps: Optional[Union[int, Dict[str, int]]] = None

    # By default, Lightning restarts the dataloader at the beginning of every epoch. If you are
    # using epoch_train_steps, you want the dataloader to persist between epochs, so set this to True.
    continuous_dataloader: bool = False

    # If True, Lightning will grab the last checkpoint of your model and resume training from
    # there - hyperparameters, model, optimizer, training state should all be resumed exactly as they left off.
    continue_training: bool = False

    # Subconfig controlling whether we should start with a pretrained model
    pretrained_model: PretrainedModelConfig

    # Whether or not to use automatic mixed precision
    precision: Precision = Precision.fp32
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    float32_matmul_precision: Matmul32Precision = Matmul32Precision.highest

    # Optional strategy passed to pl.Trainer. If n_gpus>1 and this is not set, we default to 'ddp'.
    # None = default single-process.
    strategy: Optional[str] = None

    # Controls whether training will stop early if a specified metric stops showing improvement
    # in the given number of epochs.
    early_stopping: EarlyStoppingConfig
    checkpointing: ModelCheckpointingConfig

    # how much debug/info text to print to the terminal. Common values are 0/10/20.
    verbosity: int = 20
    # how often to log things to tensorboard
    tb_log_frequency: int = 1

    # EMA
    ema: Optional[float] = None

    # Subconfigs
    data: DataLoaderConfig
    opt: OptimizerConfig
    lr: LRConfig
    profiler: ModelProfilingConfig

    # self-explanatory :)
    test: bool = False
    metric_damping: Optional[float] = None  # CURRENTLY DOES NOT WORK WITH LIGHTNING
    logger: LoggerType = LoggerType.tensorboard


class CheckpointConfig():
    best: Optional[bool] = None
    last: Optional[bool] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    version: Optional[int] = None
    ema: bool = False

class EnvironmentConfig():
    n_gpus: int = torch.cuda.device_count()
    devices: Optional[List[int]] = None
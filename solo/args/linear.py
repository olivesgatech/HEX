import os

import omegaconf
from omegaconf import OmegaConf
from solo.methods.base import BaseMethod
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import omegaconf_select
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from solo.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True

_N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    'cifar10c': 10,
    'blood': 8,
    "stl10": 10,
    'svhn':10,
    'cub':200,
    "imagenet": 1000,
    "imagenet100": 100,
    "tinyimagenet200":200,
    "INaturalist":10000,
    "pathmnist": 9,
    "bloodmnist": 8,
    "octmnist": 4,
    "dermamnist": 7,
    "mnist":10,
    'tissuemnist': 8,
    'retinamnist': 5,
    'organamnist': 11,
    'organcmnist': 11,
    'organsmnist': 11,
    'pneumoniamnist':2,
    "stanford_cars":196,
    "oxford102flowers":102,
    "fgvc_aircraft":100,
    "nabirds":1011,
    "stanford_dogs":120,
    "oxfordpets":37,
    "food101":101,
    "caltech101":102,
    "cub200":200,
    'chestmnist':2,
    "cifar100_aquatic_mammals":5,
    "cifar100_fish":5,
    "cifar100_flowers":5,
    "cifar100_trees":5,
    "cifar100_vehicle_1":5
}

_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    'cub',
    "stl10",
    'svhn',
    'blood',
    "INaturalist",
    "imagenet",
    "tinyimagenet200",
    "imagenet100",
    "pathmnist",
    "bloodmnist",
    'pneumoniamnist',
    "octmnist",
    "mnist",
    "dermamnist",
    "stanford_cars",
    "cub200",
    "oxford102flowers",
    "fgvc_aircraft",
    "nabirds",
    "stanford_dogs",
    "oxfordpets",
    'chestmnist',
    'tissuemnist',
    'retinamnist',
    'organamnist',
    'organcmnist',
    'organsmnist',
    "food101",
    "caltech101",
    "cifar100_aquatic_mammals",
    "cifar100_fish",
    "cifar100_flowers",
    "cifar100_trees",
    "cifar100_vehicle_1",
    "custom",
]


def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for dataset config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")
    assert not OmegaConf.is_missing(cfg, "data.val_path")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)

    return cfg


def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for wandb config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)

    return cfg


def add_and_assert_lightning_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for Pytorch Lightning config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.seed = omegaconf_select(cfg, "seed", 5)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg


def parse_cfg(cfg: omegaconf.DictConfig):
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for dali
    if _dali_available:
        cfg = ClassificationDALIDataModule.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

    # backbone
    assert not omegaconf.OmegaConf.is_missing(cfg, "backbone.name")
    assert cfg.backbone.name in BaseMethod._BACKBONES

    # backbone kwargs
    cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

    assert not omegaconf.OmegaConf.is_missing(cfg, "pretrained_feature_extractor")

    cfg.pretrain_method = omegaconf_select(cfg, "pretrain_method", None)

    # extra training options
    cfg.auto_augment = omegaconf_select(cfg, "auto_augment", False)
    cfg.label_smoothing = omegaconf_select(cfg, "label_smoothing", 0.0)
    cfg.mixup = omegaconf_select(cfg, "mixup", 0.0)
    cfg.cutmix = omegaconf_select(cfg, "cutmix", 0.0)

    # augmentation related (crop size and custom mean/std values for normalization)
    cfg.data.augmentations = omegaconf_select(cfg, "data.augmentations", {})
    cfg.data.augmentations.crop_size = omegaconf_select(cfg, "data.augmentations.crop_size", 224)
    cfg.data.augmentations.mean = omegaconf_select(
        cfg, "data.augmentations.mean", IMAGENET_DEFAULT_MEAN
    )
    cfg.data.augmentations.std = omegaconf_select(
        cfg, "data.augmentations.std", IMAGENET_DEFAULT_STD
    )

    # extra processing
    if cfg.data.dataset in _N_CLASSES_PER_DATASET:
        cfg.data.num_classes = _N_CLASSES_PER_DATASET[cfg.data.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        cfg.data.num_classes = max(
            1,
            sum(entry.is_dir() for entry in os.scandir(cfg.data.train_path)),
        )

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # adjust lr according to batch size
    cfg.num_nodes = omegaconf_select(cfg, "num_nodes", 1)
    scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes / 256
    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
        cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
            cfg,
            "optimizer.kwargs.exclude_bias_n_norm",
            False,
        )
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    return cfg

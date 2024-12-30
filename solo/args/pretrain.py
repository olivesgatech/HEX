import os

import omegaconf
from omegaconf import OmegaConf
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import omegaconf_select

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

_N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    'blood': 8,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
    "tinyimagenet200":200,
    "INaturalist":10000,
    "pathmnist": 9,
    "bloodmnist": 8,
    "octmnist": 4,
    'svhn':10,
    'mnist':10,
    "chestmnist": 2,
    "pneumoniamnist": 2,
    "dermamnist": 7,
    'pathmnist': 9,
    'tissuemnist': 8,
    'retinamnist': 5,
    'organamnist': 11,
    'organcmnist': 11,
    'organsmnist': 11,
    "stanford_cars":196,
    "oxford102flowers":102,
    "fgvc_aircraft":100,
    "nabirds":1011,
    "stanford_dogs":120,
    "oxfordpets":37,
    "food101":101,
    "caltech101":102,
    "cub200":200,
    "cifar100_aquatic_mammals":5,
    "cifar100_fish":5,
    "cifar100_flowers":5,
    "cifar100_trees":5,
    "cifar100_vehicle_1":5
}

_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "stl10",
    'blood',
    'mnist',
    'svhn',
    "INaturalist",
    "imagenet",
    "tinyimagenet200",
    "imagenet100",
    "pathmnist",
    "chestmnist",
    "bloodmnist",
    "pneumoniamnist",
    "octmnist",
    "dermamnist",
    "stanford_cars",
    "cub200",
    'tissuemnist',
    'retinamnist',
    'organamnist',
    'organcmnist',
    'organsmnist',
    "oxford102flowers",
    "fgvc_aircraft",
    "nabirds",
    "stanford_dogs",
    "oxfordpets",
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

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    # if validation path is not available, assume that we want to skip eval
    cfg.data.val_path = omegaconf_select(cfg, "data.val_path", None)
    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.no_labels = omegaconf_select(cfg, "data.no_labels", False)
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)
    cfg.debug_augmentations = omegaconf_select(cfg, "debug_augmentations", False)

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
    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for dali
    if _dali_available:
        cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)

    # default values for auto_umap
    if _umap_available:
        cfg = AutoUMAP.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

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

    # find number of big/small crops
    big_size = cfg.augmentations[0].crop_size
    num_large_crops = num_small_crops = 0
    for pipeline in cfg.augmentations:
        if big_size == pipeline.crop_size:
            num_large_crops += pipeline.num_crops
        else:
            num_small_crops += pipeline.num_crops
    cfg.data.num_large_crops = num_large_crops
    cfg.data.num_small_crops = num_small_crops

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # adjust lr according to batch size
    cfg.num_nodes = omegaconf_select(cfg, "num_nodes", 1)
    scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes / 256
    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
    if cfg.data.val_path is not None:
        assert not OmegaConf.is_missing(cfg, "optimizer.classifier_lr")
        cfg.optimizer.classifier_lr = cfg.optimizer.classifier_lr * scale_factor

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

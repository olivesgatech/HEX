# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import medmnist
from medmnist import INFO, Evaluator
import torch
import  random
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import STL10, ImageFolder, INaturalist
import medmnist
from medmnist import INFO, Evaluator
try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True
import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets



class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root: str, name: str,level: int,
                 transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        self.level = level
        if(self.level == -1):
            self.data = np.load(data_path)
            self.targets = np.load(target_path)
        elif (self.level == 1):
            self.data = np.load(data_path)[0:10000, :, :, :]

            self.targets = np.load(target_path)[0:10000]
        elif (self.level == 2):
            self.data = np.load(data_path)[10000:20000, :, :, :]

            self.targets = np.load(target_path)[10000:20000]
        elif (self.level == 3):
            self.data = np.load(data_path)[20000:30000, :, :, :]

            self.targets = np.load(target_path)[20000:30000]
        elif (self.level == 4):
            self.data = np.load(data_path)[30000:40000, :, :, :]

            self.targets = np.load(target_path)[30000:40000]

        elif(self.level == 5):
            self.data = np.load(data_path)[40000:,:,:,:]

            self.targets = np.load(target_path)[40000:]


    def __getitem__(self, index):

        img, targets = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)

def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }
    return pipeline


def prepare_transforms(dataset: str) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(size=32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }
    gray_color_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(size=32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }
    gray_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((.1906), (.2112)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(size=32),
                transforms.ToTensor(),
                transforms.Normalize((.1906), (.2112)),
            ]
        ),
    }
    tiny_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=64, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "cifar10c": cifar_pipeline,
        'bloodmnist': cifar_pipeline,
        'chestmnist': gray_color_pipeline,
        'svhn': cifar_pipeline,
        'dermamnist': cifar_pipeline,
        'pathmnist': cifar_pipeline,
        'caltech101': imagenet_pipeline,
        "pneumoniamnist": gray_color_pipeline,
        'tissuemnist':gray_color_pipeline,
        'organsmnist':gray_color_pipeline,
        'organamnist': gray_color_pipeline,
        'organcmnist': gray_color_pipeline,
        'retinamnist': cifar_pipeline,
        "mnist": gray_color_pipeline,
        'octmnist': gray_color_pipeline,
        "stl10": stl_pipeline,
        "tinyimagenet200":tiny_pipeline,
        'cub':imagenet_pipeline,
        'food101':imagenet_pipeline,
        'INaturalist':imagenet_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "cifar100_aquatic_mammals":cifar_pipeline,
        "cifar100_fish":cifar_pipeline,
        "cifar100_flowers":cifar_pipeline,
        "cifar100_trees":cifar_pipeline,
        "cifar100_vehicle_1":cifar_pipeline,
        "custom": custom_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    download: bool = True,
    level: int = 1,
    data_fraction: float = -1.0,
    target_type_train = 'full',
    target_type_val = 'full'
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"


    assert dataset in ["cifar10", "cifar100", "stl10", "imagenet", "imagenet100",
                       'pathmnist', "bloodmnist", "cifar100_aquatic_mammals",
                       "cifar100_fish", "cifar100_flowers", "cifar100_trees", "cifar100_vehicle_1",
                       'cub', 'food101', 'svhn', 'mnist', 'caltech101', 'chestmnist', 'octmnist', 'dermamnist',
                       "pneumoniamnist", "custom", 'cifar10c', "INaturalist", 'tinyimagenet200',
                       'tissuemnist','organamnist','organcmnist','organsmnist','retinamnist']

    if dataset == "cifar10" or dataset == "cifar100":
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            train_data_path,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            val_data_path,
            train=False,
            download=download,
            transform=T_val,
        )
    elif "cifar100_" in dataset:
        train_dataset = ImageFolder(
            train_data_path,
            transform=T_train,
        )

        val_dataset = ImageFolder(
            val_data_path,
            transform=T_val,
        )
    elif (dataset == 'bloodmnist' or dataset == 'retinamnist'
          or dataset == 'organamnist' or dataset == 'organcmnist'
          or dataset == 'organsmnist' or dataset == 'tissuemnist'):

        info = INFO[dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', transform=T_train, root=train_data_path)
        val_dataset = DataClass(split='test', transform=T_val, root=val_data_path)
    elif dataset == 'dermamnist':
        info = INFO[dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', transform=T_train, root=train_data_path)
        val_dataset = DataClass(split='test', transform=T_val, root=val_data_path)
    elif dataset == 'pneumoniamnist':
        info = INFO[dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', transform=T_train, root=train_data_path)
        val_dataset = DataClass(split='test', transform=T_val, root=val_data_path)
    elif dataset == 'chestmnist':
        info = INFO[dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', transform=T_train, root=train_data_path)
        val_dataset = DataClass(split='test', transform=T_val, root=val_data_path)
    elif dataset == 'pathmnist':
        info = INFO[dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', transform=T_train, root=train_data_path)
        val_dataset = DataClass(split='test', transform=T_val, root=val_data_path)
    elif dataset == 'octmnist':
        info = INFO[dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', transform=T_train, root=train_data_path)
        val_dataset = DataClass(split='test', transform=T_val, root=val_data_path)
    elif dataset == 'cifar10c':
        DatasetClass = vars(torchvision.datasets)['CIFAR10']
        train_dataset = DatasetClass(
            train_data_path,
            train=True,
            download=download,
            transform=T_train,
        )
        val_dataset = CIFAR10C(root=val_data_path,level=level,name=target_type_val,transform=T_val)
    elif dataset == 'mnist':
        
        train_dataset = torchvision.datasets.MNIST(
            train_data_path,
            train=True,
            download=True,
            transform=T_train,
        )

        val_dataset = torchvision.datasets.MNIST(
            val_data_path,
            train=False,
            download=True,
            transform=T_val,
        )
    elif dataset == 'svhn':
        
        train_dataset = torchvision.datasets.SVHN(
            train_data_path,
            split='train',
            download=True,
            transform=T_train,
        )

        val_dataset = torchvision.datasets.SVHN(
            val_data_path,
            split='test',
            download=True,
            transform=T_val,
        )
    elif dataset == 'caltech101':
        train_dataset = ImageFolder(
            root = train_data_path,
            transform=T_train,
        )
        val_dataset = ImageFolder(
            root = val_data_path,
            transform=T_val,
        )
    elif dataset == 'cub':
        train_dataset = ImageFolder(
            root = train_data_path,
            transform=T_train,
        )
        val_dataset = ImageFolder(
            root = val_data_path,
            transform=T_val,
        )
    elif dataset == 'food101':
        train_dataset = ImageFolder(
            root = train_data_path,
            transform=T_train,
        )
        val_dataset = ImageFolder(
            root = val_data_path,
            transform=T_val,
        )
    elif dataset == "stl10":
        train_dataset = STL10(
            train_data_path,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            val_data_path,
            split="test",
            download=download,
            transform=T_val,
        )
    elif dataset == "INaturalist":
        train_dataset = INaturalist(
            train_data_path,
            version='2021_train_mini',
            target_type = target_type_train,
            download=False,
            transform=T_train,
        )
        val_dataset = INaturalist(
            val_data_path,
            version='2021_valid',
            target_type = target_type_val,
            download=False,
            transform=T_val,
        )

    elif dataset in ["imagenet", "imagenet100", "custom",'tinyimagenet200']:
        if data_format == "h5":
            assert _h5_available
            train_dataset = H5Dataset(dataset, train_data_path, T_train)
            val_dataset = H5Dataset(dataset, val_data_path, T_val)
        else:
            train_dataset = ImageFolder(train_data_path, T_train)
            val_dataset = ImageFolder(val_data_path, T_val)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4,data_fraction=-1,
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """
    
    if(data_fraction > 0):
        #sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=int(len(train_dataset) * data_fraction))
        subset_indices = random.sample(range(0,len(train_dataset)),int(len(train_dataset) * data_fraction))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=SubsetRandomSampler(subset_indices),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    data_fraction: float = -1.0,
    auto_augment: bool = False,
    target_type_train = 'full',
    target_type_val = 'full'
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """

    T_train, T_val = prepare_transforms(dataset)
    if auto_augment:
        T_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,  # don't use color jitter when doing random aug
            auto_augment="rand-m9-mstd0.5-inc1",  # auto augment string
            interpolation="bicubic",
            re_prob=0.25,  # random erase probability
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=-1,
        target_type_train = target_type_train,
        target_type_val = target_type_val
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        data_fraction=data_fraction
    )
    return train_loader, val_loader

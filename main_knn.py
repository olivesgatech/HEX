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

import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from solo.args.knn import parse_args_knn
from solo.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
import numpy as np
from solo.methods import METHODS
from solo.utils.knn import WeightedKNNClassifier
def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    backbone_features, proj_features, labels = [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_features.append(outs["feats"].detach())
        proj_features.append(outs["z"])
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    return backbone_features, proj_features, labels


@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = knn.compute()

    # free up memory
    del knn

    return acc1, acc5


def main():
    args = parse_args_knn()

    # build paths
    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = ckpt_dir / args.ckpt

    # load ckpt_dir = Path(args.pretrained_checkpoint_dir)
    #     args_path = ckpt_dir / "args.json"
    #     ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]arguments
    with open(args_path) as f:
        method_args = json.load(f)
    cfg = OmegaConf.create(method_args)

    # build the model
    model = METHODS[method_args["method"]].load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)

    # prepare data
    _, T = prepare_transforms(args.dataset)
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_train=T,
        T_val=T,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # extract train features
    train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
    print(train_targets)
    #coarse_train_targets = sparse2coarse(train_targets.detach().cpu().numpy())
    #coarse_train_targets = torch.from_numpy(coarse_train_targets).to('cuda:0')
    train_features = {"backbone": train_features_bb, "projector": train_features_proj}

    # extract test features
    test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
    #coarse_test_targets = sparse2coarse(test_targets.detach().cpu().numpy())
    #coarse_test_targets = torch.from_numpy(coarse_test_targets).to('cuda:0')
    test_features = {"backbone": test_features_bb, "projector": test_features_proj}

    # run k-nn for all possible combinations of parameters
    for feat_type in args.feature_type:
        print(f"\n### {feat_type.upper()} ###")
        for k in args.k:
            for distance_fx in args.distance_function:
                temperatures = args.temperature if distance_fx == "cosine" else [None]
                for T in temperatures:

                    acc1, acc5 = run_knn(
                        train_features=train_features[feat_type],
                        train_targets=train_targets,
                        test_features=test_features[feat_type],
                        test_targets=test_targets,
                        k=k,
                        T=T,
                        distance_fx=distance_fx,
                    )
                    with open('knn_results.txt','a') as f:
                        f.write('True Targets' + " ")
                        f.write(args.ckpt + " ")
                        f.write(feat_type)
                        f.write("---")
                        f.write(f"Running k-NN with params: distance_fx={distance_fx}, k={k}, T={T}...")
                        f.write(f"Result: acc@1={acc1}, acc@5={acc5}")
                        f.write('\n\n')




if __name__ == "__main__":
    main()
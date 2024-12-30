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

from typing import Any, Dict, List, Sequence

import numpy as np
import omegaconf
import torch
import torch.nn as nn
from solo.losses.vicreg import vicreg_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.losses.simclr import simclr_loss_func, simclr_loss_func_hex_fixed,simclr_loss_func_superclass, simclr_loss_func_hex_adaptive

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


class VICReg(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                sim_loss_weight (float): weight of the invariance term.
                var_loss_weight (float): weight of the variance term.
                cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__(cfg)

        self.sim_loss_weight: float = cfg.method_kwargs.sim_loss_weight
        self.var_loss_weight: float = cfg.method_kwargs.var_loss_weight
        self.cov_loss_weight: float = cfg.method_kwargs.cov_loss_weight

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(VICReg, VICReg).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")

        cfg.method_kwargs.sim_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.sim_loss_weight",
            25.0,
        )
        cfg.method_kwargs.var_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.var_loss_weight",
            25.0,
        )
        cfg.method_kwargs.cov_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.cov_loss_weight",
            1.0,
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """
        indexes = batch[0]
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]
        z_cat = torch.cat(out["z"])

        # ------- vicreg loss -------
        if(self.cfg.type == 'standard'):
            vicreg_loss = vicreg_loss_func(
                z1,
                z2,
                sim_loss_weight=self.sim_loss_weight,
                var_loss_weight=self.var_loss_weight,
                cov_loss_weight=self.cov_loss_weight,
            )
        elif(self.cfg.type == 'vicreg_simclr'):
            vicreg_loss = simclr_loss_func(
                z_cat,
                indexes=indexes,
                temperature=self.cfg.temperature,
            ) + vicreg_loss_func(z1, z2, self.sim_loss_weight, self.var_loss_weight, self.cov_loss_weight)
            
        elif(self.cfg.type == 'HEX_fixed'):
            if (self.cfg.step_type == 'step'):
                if (self.current_epoch % self.cfg.epoch_step == 0 and batch_idx == 0 and self.current_epoch != 0):
                    self.cfg.distance_thresh = self.cfg.distance_thresh - .1
            vicreg_loss = self.cfg.alpha * simclr_loss_func_hex_fixed(
                z_cat,
                indexes=indexes,
                temperature=self.cfg.temperature,
                distance_thresh=self.cfg.distance_thresh,
                beta=self.cfg.beta,
                tau_plus=self.cfg.tau_plus
            ) + (1-self.cfg.alpha) * vicreg_loss_func(z1, z2, self.sim_loss_weight, self.var_loss_weight, self.cov_loss_weight)
        elif(self.cfg.type == 'Hex_adaptive'):
            vicreg_loss = self.cfg.alpha * 5 * simclr_loss_func_hex_adaptive(
                z_cat,
                indexes=indexes,
                temperature=self.cfg.temperature,
                beta=self.cfg.beta,
                tau_plus=self.cfg.tau_plus,
                current_epoch = self.current_epoch,
            ) + (1-self.cfg.alpha) * vicreg_loss_func(z1, z2, self.sim_loss_weight, self.var_loss_weight, self.cov_loss_weight)
        else:
            targets_og = batch[-1]
            n_augs = self.num_large_crops + self.num_small_crops
            super_targets = sparse2coarse(targets_og.detach().cpu().numpy())
            super_targets = torch.from_numpy(super_targets).to('cuda:0')

            vicreg_loss = simclr_loss_func_superclass(
                z_cat,
                indexes_class=indexes,
                indexes_super=super_targets,
                temperature=self.cfg.temperature,
            ) +  vicreg_loss_func(z1, z2, self.sim_loss_weight, self.var_loss_weight, self.cov_loss_weight)

        self.log("train_vicreg_loss", vicreg_loss, on_epoch=True, sync_dist=True)

        return vicreg_loss + class_loss
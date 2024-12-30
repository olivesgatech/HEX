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

import torch
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank
import numpy as np
import torch.nn as nn

def simclr_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss









def simclr_loss_func_hex_fixed(z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1, distance_thresh: float = .9,beta:float=1,tau_plus:float=1) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    out = z
    device='cuda:0'
    batch_size = int(z.shape[0]/2)
    distance_thresh = distance_thresh
    tau_plus = tau_plus
    beta = beta

    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    anchor_count = 2
    contrast_count = 2
    cosine_similarity = torch.matmul(out, out.T)
    anchor_dot_contrast = torch.div(
        torch.matmul(out, out.T),
        temperature)

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)

    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )

    f_x = cosine_similarity
    f_x -= f_x.min(1, keepdim=True)[0]
    f_x /= f_x.max(1, keepdim=True)[0]
    feature_mask = f_x > distance_thresh
    feature_mask = feature_mask + mask
    feature_mask = feature_mask * logits_mask
    feature_mask = feature_mask > distance_thresh
    super_mask = feature_mask.long()
    super_logits = super_mask * logits
    mask = mask * logits_mask

    super_mask = super_mask * logits_mask
    neg_mask = logits_mask - super_mask

    # Super Negatives Only
    exp_logits_super = torch.exp(logits) * super_mask
    # Full Negatives without the SuperClass Negatives
    exp_logits = torch.exp(logits) * (neg_mask)
    # compute mean of log-likelihood over positive

    pos_matrix = torch.exp((mask * logits).sum())

    neg_log = torch.log(exp_logits_super.sum(1, keepdim=True))

    temperature = temperature
    base_temperature = temperature

    tau_plus = tau_plus
    N = batch_size * 2 - 2
    imp = (beta * neg_log).exp()
    reweight_neg = (imp * super_logits).sum(dim=-1) / imp.mean(dim=-1)
    Ng = (-tau_plus * N * pos_matrix + reweight_neg) / (1 - tau_plus)
    Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    log_prob = logits - Ng - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

def simclr_loss_func_hex_adaptive(z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1, beta:float=1,tau_plus:float=1,current_epoch = 0,warm_epoch = 10) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    out = z
    device='cuda:0'
    batch_size = int(z.shape[0]/2)
    tau_plus = tau_plus
    beta = beta

    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    anchor_count = 2
    contrast_count = 2
    cosine_similarity = torch.matmul(out, out.T)
    square_mag = torch.diag(cosine_similarity)
    inv_square_mag = 1/square_mag
    inv_square_mag[torch.isinf(inv_square_mag)] = 0
    inv_mag = torch.sqrt(inv_square_mag)
    cosine = cosine_similarity * inv_mag
    cosine = cosine.T * inv_mag
    
    
    cosine_average = torch.mean(torch.quantile(cosine,0.99,dim=1,interpolation='nearest'))
    anchor_dot_contrast = torch.div(
        torch.matmul(out, out.T),
        temperature)

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)

    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )

    f_x = cosine
    #f_x -= f_x.min(1, keepdim=True)[0]
    #f_x /= f_x.max(1, keepdim=True)[0]
    feature_mask = f_x > cosine_average
    feature_mask = feature_mask + mask
    feature_mask = feature_mask * logits_mask
    feature_mask = feature_mask > cosine_average
    super_mask = feature_mask.long()
    super_logits = super_mask * logits
    mask = mask * logits_mask

    super_mask = super_mask * logits_mask
    neg_mask = logits_mask - super_mask

    # Super Negatives Only
    exp_logits_super = torch.exp(logits) * super_mask
    # Full Negatives without the SuperClass Negatives
    exp_logits = torch.exp(logits) * (neg_mask)
    # compute mean of log-likelihood over positive

    pos_matrix = torch.exp((mask * logits).sum())

    neg_log = torch.log(exp_logits_super.sum(1, keepdim=True))

    temperature = temperature
    base_temperature = temperature

    tau_plus = tau_plus
    N = batch_size * 2 - 2
    imp = (beta * neg_log).exp()
    reweight_neg = (imp * super_logits).sum(dim=-1) / imp.mean(dim=-1)
    Ng = (-tau_plus * N * pos_matrix + reweight_neg) / (1 - tau_plus)
    Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    log_prob = logits - Ng - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss



def simclr_loss_func_superclass(z: torch.Tensor, indexes_class: torch.Tensor,indexes_super: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    out = z
    device='cuda:0'
    batch_size = int(z.shape[0]/2)
    tau_plus = .1
    beta = 1

    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    super_labels = indexes_super.reshape(len(indexes_super), 1)
    super_mask = torch.eq(super_labels, super_labels.T).float().to(device)
    anchor_count = 2
    contrast_count = 2
    cosine_similarity = torch.matmul(out, out.T)
    anchor_dot_contrast = torch.div(
        torch.matmul(out, out.T),
        temperature)

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    super_mask = super_mask.repeat(anchor_count, contrast_count)
    super_logits = super_mask * logits
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    super_mask = super_mask * logits_mask
    neg_mask = logits_mask - super_mask

    # Super Negatives Only
    exp_logits_super = torch.exp(logits) * super_mask
    # Full Negatives without the SuperClass Negatives
    exp_logits = torch.exp(logits) * (neg_mask)
    # compute mean of log-likelihood over positive
    pos_matrix = torch.exp((mask * logits).sum())

    neg_log = torch.log(exp_logits_super.sum(1, keepdim=True))



    N = batch_size * 2 - 2
    imp = (beta * neg_log).exp()
    reweight_neg = (imp * super_logits).sum(dim=-1) / imp.mean(dim=-1)
    Ng = (-tau_plus * N * pos_matrix + reweight_neg) / (1 - tau_plus)
    Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    log_prob = logits - Ng - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    temperature = temperature
    base_temperature = temperature
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss
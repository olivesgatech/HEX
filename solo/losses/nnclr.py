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

def nnclr_loss_func(nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
    predicted features p from view 2.

    Args:
        nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
        p (torch.Tensor): NxD Tensor containing predicted features from view 2
        temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
            to 0.1.

    Returns:
        torch.Tensor: NNCLR loss.
    """

    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)
    # to be consistent with simclr, we now gather p
    # this might result in suboptimal results given previous parameters.
    p = gather(p)

    logits = nn @ p.T / temperature

    rank = get_rank()
    n = nn.size(0)
    labels = torch.arange(n * rank, n * (rank + 1), device=p.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def nnclr_loss_func_hard_decompose(nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.1,distance_thresh: float=0.9) -> torch.Tensor:
    """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
    predicted features p from view 2.

    Args:
        nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
        p (torch.Tensor): NxD Tensor containing predicted features from view 2
        temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
            to 0.1.

    Returns:
        torch.Tensor: NNCLR loss.
    """

    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)
    out = torch.cat([nn, p], dim=0)
    device = 'cuda:0'
    batch_size = 256
    distance_thresh = distance_thresh
    tau_plus = .1
    beta = 1
    # to be consistent with simclr.yaml, we now gather p
    # this might result in suboptimal results given previous parameters.
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




def nnclr_loss_func_decompose_adaptive(nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.1,current_epoch = 0,warm_epoch = 10) -> torch.Tensor:
    """
    Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
    predicted features p from view 2.

    Args:
        nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
        p (torch.Tensor): NxD Tensor containing predicted features from view 2
        temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
            to 0.1.

    Returns:
        torch.Tensor: NNCLR loss.
    """

    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)
    out = torch.cat([nn, p], dim=0)
    device = 'cuda:0'
    batch_size = 256
    
    tau_plus = .1
    beta = 1
    # to be consistent with simclr.yaml, we now gather p
    # this might result in suboptimal results given previous parameters.
    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    
    
    anchor_count = 2
    contrast_count = 2
    cosine_similarity = torch.matmul(out, out.T)
    cosine_similarity = cosine_similarity.double()
    
    
    square_mag = torch.diag(cosine_similarity)
    inv_square_mag = 1/square_mag
    inv_square_mag[torch.isinf(inv_square_mag)] = 0
    inv_mag = torch.sqrt(inv_square_mag)
    cosine = cosine_similarity * inv_mag
    cosine = cosine.T * inv_mag
    
    
    cosine_average = torch.mean(torch.quantile(cosine,.99,dim=1,interpolation='nearest'))
    if(current_epoch <= warm_epoch):
        cosine_average = .9
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



def nnclr_loss_func_superclass(nn: torch.Tensor, p: torch.Tensor,indexes_super: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
    predicted features p from view 2.

    Args:
        nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
        p (torch.Tensor): NxD Tensor containing predicted features from view 2
        temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
            to 0.1.

    Returns:
        torch.Tensor: NNCLR loss.
    """

    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)
    out = torch.cat([nn, p], dim=0)
    device = 'cuda:0'
    batch_size = 256
    tau_plus = .1
    beta = 1
    # to be consistent with simclr.yaml, we now gather p
    # this might result in suboptimal results given previous parameters.
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
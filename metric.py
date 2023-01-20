import torch


def batched_cdist_l2(x1, x2):
    """Fast implementation of batched pairwise L2"""
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res


def euclidean_distance(prototypes, query):
    """Calculates the distance between given prototypes and query tensor using euclidean"""
    dist = torch.squeeze(torch.cdist(prototypes[None].flatten(2), torch.unsqueeze(query, dim=0)))
    return dist


def cosine_distance(prototype, query):
    """Calculates the distance between given prototypes and query tensor using cosine"""
    dist = 0
    return dist


def kl_divergence(prototypes, query):
    """Calculates the distance between given prototypes and query tensor using KL divergence"""
    dist = torch.squeeze(torch.distributions.kl_divergence(prototypes[None].flatten(2), torch.unsqueeze(query, dim=0)))
    return dist


"""
MIT License

Copyright (c) 2023 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
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


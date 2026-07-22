# Some methods adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(t, dim=1):
    return F.normalize(t, dim=dim, eps=1e-5, p=2)

@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


class EntropyMaxLoss(nn.Module):
    def __init__(self, cfg):
        super(EntropyMaxLoss, self).__init__()
        self.cfg = cfg

    def forward(self, kernel, samples, **kwargs):
        # The feature/kernel memory bank is intentionally empty during the
        # first training steps.  In that warm-up period ``train.py`` passes
        # ``None`` for ``samples`` (and may also pass ``None`` for ``kernel``),
        # so this optional auxiliary loss must be a differentiable no-op
        # rather than trying to take ``len(None)``.
        if kernel is None or samples is None:
            reference = kernel if kernel is not None else samples
            if reference is None:
                # No tensor is available to anchor the graph.  A scalar zero
                # is sufficient; the wrapper multiplies it by this loss's
                # configured weight.
                loss = torch.tensor(0.0)
            else:
                loss = reference.sum() * 0.0
            return [
                {
                    'name': 'loss_kernel_align',
                    'loss': loss,
                    'weight': self.cfg.weight
                }
            ]

        M, d, ks = kernel.shape[:3]
        N = len(samples)
        if M == 0 or N == 0:
            return [{
                'name': 'loss_kernel_align',
                'loss': kernel.sum() * 0.0,
                'weight': self.cfg.weight
            }]
        kernel = kernel.reshape(M, d*ks*ks)
        samples = samples.reshape(N, d*ks*ks)

        kernel = F.normalize(kernel, p=2, dim=1)
        samples = F.normalize(samples, p=2, dim=1)
        sim = kernel @ samples.transpose(1,0) # M * N
        logp = torch.log_softmax(sim / self.cfg.temperature, dim=1)
        p = torch.softmax(sim / self.cfg.temperature, dim=1)
        nll = -(p*logp).sum(-1)
        return [
            {
                'name': 'loss_kernel_align',
                'loss': nll.mean(),
                'weight': self.cfg.weight
            }
        ]

class KernelAlignLoss(nn.Module):

    def __init__(self, cfg):
        super(KernelAlignLoss, self).__init__()
        self.cfg = cfg
        self.rbf = RBF()

    def forward(self, kernel, kernel_target, **kwargs):
        X, Y = kernel, kernel_target
        
        N_y, d, ks = Y.shape[:3]
        D = d*ks*ks
        N_x = X.shape[0]
        X = X.reshape(N_x, D)
        Y = Y.reshape(N_y, D)
        
        K = self.rbf(torch.vstack([X, Y]))
        XX = K[:N_x, :N_x].mean()
        XY = K[:N_x, N_x:].mean()
        YY = K[N_x:, N_x:].mean()
        loss = XX - 2 * XY + YY

        return [
            {
                'name': 'loss_kernel_align',
                'loss': loss.mean(),
                'weight': self.cfg.weight
            }
        ]


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = None
        self.bandwidth = bandwidth
        self.n_kernels = n_kernels
        self.mul_factor = mul_factor

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        if self.bandwidth_multipliers is None:
            self.bandwidth_multipliers = self.mul_factor ** \
                (torch.arange(self.n_kernels, device=X.device) - self.n_kernels // 2)

        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)

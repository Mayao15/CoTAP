from numpy import nonzero
import torch
import time

from utils import *
import torch.nn.functional as F


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        if n_classes == 21:
            n_classes -= 1
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, sal=None, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            # cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
            #     .permute(0, 3, 1, 2).to(torch.float32)
            cls_pred = torch.argmax(inner_products, dim=1)
            n_cls = self.n_classes
            if sal is not None:
                cls_pred += 1
                cls_pred = (cls_pred * sal.squeeze(1)).long()
                n_cls += 1

            cluster_probs = F.one_hot(cls_pred, n_cls).permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        if sal is not None:
            cluster_probs_tmp = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]).permute(0, 3, 1, 2).to(torch.float32)
            cluster_loss = -(cluster_probs_tmp * inner_products * sal).sum(1).mean()
        else:
            cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    if len(a.shape) == 4:
        return torch.einsum("nchw,ncij->nhwij", a, b)
    elif len(a.shape) == 3:
        return torch.einsum("nci,ncj->nij", a, b)
    else:
        assert False


def sample(t: torch.Tensor, coords: torch.Tensor):
    if len(coords.shape) == 4:
        return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)
    elif len(coords.shape) == 3:
        return F.grid_sample(t, coords.unsqueeze(1), padding_mode='border', align_corners=True).squeeze(1)
    else:
        assert False

@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])

def print_tensor(x, name=''):
    print('%s: '%name, x.shape, x.max().item(), x.min().item(), x.mean().item(), x.median().view(-1).item(), x.dtype)

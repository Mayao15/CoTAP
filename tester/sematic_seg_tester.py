import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
import numpy as np
import faiss 

from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric


class SematicSegTester(nn.Module):
    def __init__(self, cfg, name):
        super().__init__()
        self.cfg = cfg
        self.name = name
        n_classes = self.cfg.n_classes
        self.linear_metrics = UnsupervisedMetrics(self.name + '/', n_classes, cfg.ignore_classes, 1, False)
        # self.cluster_metrics = UnsupervisedMetrics(self.name + '/cluster_', n_classes, 1, True)

    def start(self, rank):
        ## setup probe
        # self.probe_cluster = MiniBatchKMeans(self.cfg)
        self.probe_linear = LinearMapping(self.cfg, rank, self.cfg.lr)
        # self.cluster_metrics.reset()
        self.linear_metrics.reset()

    def finetune_step(self, feats_patch, label, pl_module, sal=None, **kwargs):
        if self.cfg.include_stuff:
            sal = None
        if sal is not None:
            sal = F.interpolate(sal.cuda().float().unsqueeze(1), feats_patch.shape[-2:], mode='nearest')

        # self.probe_cluster.update(feats_patch, sal, pl_module)

        return self.probe_linear.update(feats_patch, pl_module, label=label)

    @torch.no_grad()
    def validation_step(self, feats, label, pl_module):
        # max_upsampleing_batch = 4
        # idx = 0
        # feats_upsampled = []
        # while idx < len(feats):
        #     feats_upsampled.append(F.interpolate(feats[idx:idx+max_upsampleing_batch], label.shape[-2:], mode='bilinear', align_corners=False))
        #     idx += max_upsampleing_batch
        # feats_upsampled = torch.cat(feats_upsampled, dim=0)

        preds_linear = self.probe_linear.predict(feats, pl_module=pl_module)
        # print(preds_linear.mean().item())

        preds_linear = F.interpolate(preds_linear, label.shape[-2:], mode='bilinear', align_corners=False)
        self.linear_metrics.update(preds_linear.argmax(1), label)

        # preds_cluster = self.probe_cluster.predict(feats_upsampled, pl_module=pl_module).argmax(1)
        # self.cluster_metrics.update(preds_cluster, label)

        return preds_linear

    def end(self, **kwargs):
        pass

    @torch.no_grad()
    def compute(self, outputs, **kwargs) -> None:
        tb_metrics = {
            **self.linear_metrics.compute(),
        }

        del self.probe_linear

        return tb_metrics


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, ignore_classes: list, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.ignore_classes = ignore_classes
        self.extra_clusters = extra_clusters if compute_hungarian else 0
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]

            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        iou = iou[~torch.isnan(iou)]
        iou_keep = []
        for i in range(len(iou)):
            if i not in self.ignore_classes:
                iou_keep.append(iou[i])
        iou = torch.tensor(iou_keep)

        metric_dict = {self.prefix + "mIoU": iou.mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}


class MiniBatchKMeans(nn.Module):
    def __init__(self, args: dict):
        super(MiniBatchKMeans, self).__init__()
        self.args = args
        self.centroids = None
        self.reset()

    def reset(self):
        self.faiss_module = get_faiss_module(self.args.dim)
        # self.data_count   = np.zeros(self.args.num_classes)
        if self.args.num_classes == 21:
            self.num_classes = 20
        else:
            self.num_classes = self.args.num_classes
        self.data_count   = torch.zeros(
            self.num_classes + self.args.extra_clusters).cuda()
        self.featslist    = []
        self.sallist = []
        self.num_batches = 0
        self.first = True
        self.sync = True

        if self.centroids is not None:
            del self.centroids
            self.centroids = None

    @torch.no_grad()
    def update(self, feats, sal, pl_module=None, **kwargs):
        # Normalize.
        d = feats.shape[1]

        feats = feats.permute(0,2,3,1).reshape(-1, d).detach()
        if sal is not None:
            sal = sal.reshape(-1, 1)

        self.sync = False
        self.featslist.append(feats)
        self.sallist.append(sal)
        self.num_batches += 1

        if self.num_batches == self.args.update_batches:
            self.compute_centroids(pl_module)

    def compute_centroids(self, pl_module):
        if self.sync or len(self.featslist) == 0:
            return
        self.sync = True

        faiss_module = self.faiss_module
        d = self.featslist[0].shape[-1]
        featslist = torch.cat(self.featslist)

        if self.sallist[0] is not None:
            sal = torch.cat(self.sallist)
        else:
            sal = None

        if self.first:
            # Compute initial centroids. 
            # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
            featslist = pl_module.all_gather(featslist).view(-1, d)
            if sal is not None:
                sal = pl_module.all_gather(sal).view(-1)
                featslist = featslist[sal > 0]
            self.centroids = torch.zeros(
                self.num_classes + self.args.extra_clusters,
                self.args.dim
            ).float().cuda()

            if pl_module.global_rank == 0:
                featslist_zero_center = featslist
                featslist_zero_center = F.normalize(featslist_zero_center, dim=1)

                self.centroids = get_init_centroids(
                    self.args,
                    self.num_classes + self.args.extra_clusters,
                    featslist_zero_center.cpu().numpy().astype('float32'),
                    faiss_module
                )
                self.centroids = torch.from_numpy(self.centroids).cuda()

            self.centroids = pl_module.broadcast(self.centroids, 0)
            featslist = F.normalize(featslist, dim=1)
            centroids = F.normalize(self.centroids, dim=1)
            sim = torch.mm(featslist, centroids.t())
            I = sim.argmax(-1)
            self.data_count += torch.bincount(I, minlength=len(self.data_count))
            self.data_count = pl_module.reduce(self.data_count)

            self.first = False

        else:
            if sal is not None:
                featslist = featslist[sal.view(-1) > 0]

            featslist = F.normalize(featslist, dim=1)
            centroids = F.normalize(self.centroids, dim=1)
            sim = torch.mm(featslist, centroids.t())
            I = sim.argmax(-1)
            cnt = torch.bincount(I, minlength=len(self.data_count))

            cnt = pl_module.reduce(cnt)
            self.data_count += cnt

            centroid_lr = (cnt / (self.data_count + 1e-6)).view(-1, 1)
            sum_feats_per_cls = [torch.index_select(featslist, 0, I.eq(k).nonzero().view(-1)).sum(0) \
                 for k in range(len(self.data_count))]
            sum_feats_per_cls = torch.stack(sum_feats_per_cls)
            sum_feats_per_cls = pl_module.reduce(sum_feats_per_cls)
            mean_feats_per_cls = sum_feats_per_cls / (cnt.view(-1,1) + 1e-6)

            self.centroids = (1 - centroid_lr) * centroids + centroid_lr * mean_feats_per_cls
            self.centroids = pl_module.broadcast(self.centroids, 0)

        # Empty. 
        self.featslist   = []
        self.sallist = []
        self.num_batches = 0

    def get_centroids(self, pl_module=None):
        self.compute_centroids(pl_module)
        if not self.sync and pl_module is not None:
            self.centroids = pl_module.broadcast(self.centroids, src=0)
            self.sync = True

        return self.centroids

    @torch.no_grad()
    def predict(self, x, sal=None, alpha=None, max_num=1000, pl_module=None):

        self.get_centroids(pl_module)

        normed_clusters = F.normalize(self.centroids, dim=1)
        normed_features = F.normalize(x, dim=1)

        inner_products = []
        idx = 0

        while idx < len(normed_features):
            inner_products.append(torch.einsum("bchw,nc->bnhw", \
                normed_features[idx:idx+max_num], normed_clusters))
            idx += max_num
        inner_products = torch.cat(inner_products, dim=0)

        if alpha is None:
            cls_pred = torch.argmax(inner_products, dim=1)
            n_cls = self.centroids.shape[0]
            if sal is not None:
                cls_pred += 1
                cls_pred = (cls_pred * sal.squeeze(1)).long()
                n_cls += 1

            cluster_probs = F.one_hot(cls_pred, n_cls).permute(0, 3, 1, 2).to(torch.float32)
        else:
            assert False

        return cluster_probs

def get_init_centroids(args, K, featlist, index):
    # from torch_cluster import fps
    clus = faiss.Clustering(args.dim, K)
    clus.seed  = args.seed
    clus.niter = args.kmeans_n_iter
    clus.max_points_per_centroid = 4000000
    clus.train(featlist, index)

    return faiss.vector_float_to_array(clus.centroids).reshape(K, args.dim)

def get_faiss_module(dim):
    res = faiss.StandardGpuResources()
    res.setTempMemory(128 * 1024 * 1024)
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device     = 0 #NOTE: Single GPU only. 
    # cfg.metric_type = faiss.METRIC_INNER_PRODUCT
    idx = faiss.GpuIndexFlatIP(res, dim, cfg)

    return idx


class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.bn1 = nn.SyncBatchNorm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.bn2 = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class LinearMapping(nn.Module):
    def __init__(self, args: dict, local_rank, lr):
        super(LinearMapping, self).__init__()
        self.args = args
        self.local_rank = local_rank
        self.lr = lr
        self.reset()

    def reset(self):
        local_rank = self.local_rank
        # np.random.seed(0)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        # torch.cuda.manual_seed_all(0)
        model = nn.Conv2d(self.args.dim, self.args.num_classes, (1, 1))
        # model = nn.Sequential(
        #     UpsampleBlock(self.args.dim, 512), 
        #     UpsampleBlock(512, 256), 
        #     UpsampleBlock(256, 128), 
        #     UpsampleBlock(128, 64),
        #     nn.Conv2d(64, self.args.num_classes, (1, 1))
        # )
        model.cuda()
        self.model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=100100,
            eta_min=1e-4
        )

    @torch.enable_grad()
    def update(self, feats, pl_module=None, label=None):
        
        logit = self.model(feats.data)
        logit = F.interpolate(logit, label.shape[-2:], mode='bilinear', align_corners=False)
        logit = logit.permute(0,2,3,1).reshape(-1, self.args.num_classes)
        label = label.view(-1)

        # print("\n\n", logit.shape, label, "\n\n")

        loss = self.loss_fn(logit, label)
        loss = loss.mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.lr_scheduler.step()
        
        return logit, loss

    @torch.no_grad()
    def predict(self, x, max_num=1000, pl_module=None):
        return self.model(x)

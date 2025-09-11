import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing

from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric


class ImageClsTester(nn.Module):
    def __init__(self, cfg, name):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.linear_metrics = TopAccuracyMetrics(self.name + '/linear_')
        self.knn_metrics = TopKNNMetrics(self.name + '/knn_')

    def start(self, rank):
        ## setup probe
        self.probe_linear = LinearMapping(self.cfg, rank, self.cfg.lr)
        self.knn_metrics.reset()
        self.linear_metrics.reset()
    
    def finetune_step(self, feats_cls, label, pl_module, **kwargs):
        self.probe_linear.update(feats_cls, pl_module, label=label)
        self.knn_metrics.train_features.append(feats_cls.data.cpu())
        self.knn_metrics.train_targets.append(label.data.cpu())

    @torch.no_grad()
    def validation_step(self, feats, label, pl_module):
        logits_linear = self.probe_linear.predict(feats, pl_module=pl_module)
        self.linear_metrics.update(logits_linear, label)
        self.knn_metrics.update(feats, label)

        return None

    def end(self, pl_module):
        self.knn_metrics.end(pl_module)

    @torch.no_grad()
    def compute(self, outputs, pl_module) -> None:
        tb_metrics = {
            **self.linear_metrics.compute(pl_module=pl_module),
            **self.knn_metrics.compute(pl_module=pl_module)
        }

        del self.probe_linear

        return tb_metrics
    
    def __str__(self) -> str:
        return 'ImageClsTester'


class TopKNNMetrics(Metric):
    def __init__(self, prefix: str, 
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.prefix = prefix
        self.features = []
        self.targets = []
        self.train_features = []
        self.train_targets = []

    @torch.no_grad()
    def update(self, feature: torch.Tensor, target: torch.Tensor):
        self.features.append(feature.data.cpu())
        self.targets.append(target.data.cpu())

    def end(self, pl_module):
        if isinstance(self.train_targets, list):
            train_features = torch.cat(self.train_features, dim=0)
            train_targets = torch.cat(self.train_targets, dim=0)
            train_features = pl_module.all_gather(train_features.cuda()).view(-1, train_features.shape[-1])
            train_targets = pl_module.all_gather(train_targets.cuda()).view(-1)
            self.train_features = train_features
            self.train_targets = train_targets

    def compute(self, pl_module=None):
        features = torch.cat(self.features, dim=0)
        targets = torch.cat(self.targets, dim=0)
        features = pl_module.all_gather(features.cuda()).view(-1, features.shape[-1])
        targets = pl_module.all_gather(targets.cuda()).view(-1)

        train_features = self.train_features.to(features.device)
        train_targets = self.train_targets.to(features.device)

        train_features = F.normalize(train_features, dim=1, p=2)
        features = F.normalize(features, dim=1, p=2)

        best_top1, best_params = 0, {}
        for k in [5, 10, 20, 50, 100]:
            for temp in [0.05, 0.07, 0.1, 0.15, 0.2]:
                top1, top5 = knn_classifier(
                    train_features,
                    train_targets,
                    features,
                    targets,
                    k=k,
                    T=temp
                )

                if top1 > best_top1:
                    best_top1 = top1
                    best_params = [k, temp]


        return {self.prefix + "Accuracy": best_top1}


    def reset(self):
        del self.features
        del self.targets
        del self.train_features
        del self.train_targets
        self.features = []
        self.targets = []
        self.train_features = []
        self.train_targets = []

    def __str__(self) -> str:
        return 'TopKNNMetrics'


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    if len(train_features) > 30000:
        num_chunks = 100
    else:
        num_chunks = 10
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5



class TopAccuracyMetrics(Metric):
    def __init__(self, prefix: str, 
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.prefix = prefix
        self.logits = []
        self.targets = []

    @torch.no_grad()
    def update(self, logit: torch.Tensor, target: torch.Tensor):
        self.logits.append(logit.data)
        self.targets.append(target.data)

    def compute(self, k=1, chunk=10, pl_module=None):
        logits = torch.cat(self.logits, dim=0)
        targets = torch.cat(self.targets, dim=0)
        logits = pl_module.all_gather(logits).view(-1, logits.shape[-1])
        targets = pl_module.all_gather(targets).view(-1)

        preds_topk = torch.topk(logits, k=k, dim=1)[1]
        state = ((targets.unsqueeze(1) == preds_topk).sum(1) > 0)

        metric_dict = {self.prefix + "Accuracy": (state.sum() / len(state)).item()}
        return {k: 100 * v for k, v in metric_dict.items()}

    def __str__(self) -> str:
        return 'TopAccuracyMetrics'

    def reset(self):
        del self.logits
        del self.targets
        self.logits = []
        self.targets = []

class LinearMapping(nn.Module):
    def __init__(self, args: dict, local_rank, lr):
        super(LinearMapping, self).__init__()
        self.args = args
        self.local_rank = local_rank
        self.lr = lr
        self.reset()

    def reset(self):
        local_rank = self.local_rank
        self.model = nn.Linear(self.args.dim, self.args.num_classes)
        self.model.cuda()
        self.dist_model = DistributedDataParallel(
            self.model, device_ids=[local_rank], output_device=local_rank)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)

    @torch.enable_grad()
    def update(self, feats, pl_module=None, label=None):
        logit = self.dist_model(feats.data)
        label = label.view(-1)
        loss = self.loss_fn(logit, label)
        loss = loss.mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    @torch.no_grad()
    def predict(self, x, max_num=1000, pl_module=None):
        return self.model(x)

    def __str__(self) -> str:
        return 'TopAccuracyMetrics'

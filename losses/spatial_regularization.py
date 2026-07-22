import torch.nn as nn

from models.feature_regularization import TotalVariation


class SpatialRegularizationLoss(nn.Module):
    """TV loss activated by the model's OAF-replacement selector."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.method = getattr(cfg, "feature_regularization", "oaf")
        self.weight = getattr(cfg, "weight", 0.01)
        self.tv = TotalVariation(reduction=getattr(cfg, "reduction", "mean"))

    def forward(self, outputs_stu, **kwargs):
        if self.method != "total_variation" or self.weight <= 0:
            return []
        key = "feat_%d" % self.cfg.head_idx_patch
        if key not in outputs_stu:
            return []
        return [{
            "name": "total_variation",
            "loss": self.tv(outputs_stu[key]),
            "weight": self.weight,
        }]

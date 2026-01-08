from genericpath import exists
import sys
import os
import os.path as osp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
import torchvision
import pytorch_lightning as pl
import hydra
import random
import warnings
warnings.filterwarnings("ignore")

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import cosine_scheduler, prep_args, share_cfg, load_checkpoint
from datasets.data_wrapper import DataWrapper
# from datasets.transforms import GaussianBlur
from tester.sematic_seg_tester import UnsupervisedMetrics


torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32=True


class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim//2, 1)
        self.bn1 = nn.SyncBatchNorm(in_dim//2)
        self.conv2 = nn.Conv2d(in_dim//2, in_dim, 3, 1, 1)
        self.bn2 = nn.SyncBatchNorm(in_dim)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x + input)
        x0 = self.conv3(x)
        x0 = F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=False)
        return x0


class FinetuneSegmenter(pl.LightningModule):
    def __init__(self, train_set, val_set, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_val = cfg.dataset_val['dataset_seg']
        self.automatic_optimization = False

        self.build_model(cfg.model)
        self.train_set = train_set
        self.val_set = val_set
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.linear_metrics = UnsupervisedMetrics('', self.cfg_val.n_classes, self.cfg_val.ignore_classes, 0, False)

        self.checked = False
        self.save_hyperparameters(ignore=['train_set', 'val_set', 'finetune_set', 'tester', 'loss_fn'])

        self.test_step = self.validation_step
        self.test_epoch_end = self.validation_epoch_end
        self.test_dataloader = self.val_dataloader
        self.on_test_epoch_start = self.on_validation_epoch_start
        self.best_miou = 0

    def broadcast(self, x, src):
        return self.all_gather(x)[src]

    def reduce(self, x, mode='sum'):
        if mode == 'sum':
            return self.all_gather(x).sum(0)
        else:
            return self.all_gather(x).mean(0)

    def world_size(self):
        return self.cfg.training.num_gpus

    def build_model(self, cfg):
        dim = cfg.dim
        if cfg.arch == "dino":
            from models.model_wrapper import DinoFeaturizer
            self.net = DinoFeaturizer(dim, cfg, require_grad=False, load_pretrain=False, pretrained_weights=cfg.pretrained_weights)
            del self.net.model.projection_heads
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))
        load_checkpoint(self.net, self.cfg.basic.resume_test, self.cfg.model.vit_type)

        if self.cfg_val.head == 'fcn':
            self.head = nn.Sequential(
                UpsampleBlock(dim, 512),
                UpsampleBlock(512, 384),
                UpsampleBlock(384, 256),
                nn.Dropout(0.1),
                nn.Conv2d(256, self.cfg_val.n_classes, (1, 1))
            )
        elif self.cfg_val.head == 'linear':
            self.head = nn.Conv2d(dim, self.cfg_val.n_classes, (1, 1))

    def build_loss_fn(self, cfg):
        from losses import LossWrapper
        self.loss_fn = LossWrapper(cfg)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            self.cfg_val.ft_batchsize // self.cfg.training.num_gpus,
            shuffle=True, 
            drop_last=True,
            num_workers=self.cfg.basic.num_workers, 
            pin_memory=True,
            persistent_workers=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            self.cfg_val.batchsize,
            shuffle=False,
            num_workers=self.cfg.basic.num_workers,
            pin_memory=False,
            persistent_workers=True
        )
        return val_loader

    @torch.no_grad()
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net.get_test_features(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        self.net.eval()
        self.head.train()

        batch = batch[0]
        imgs = batch['img']
        lbls = batch['label']
        
        with torch.no_grad():
            embs = self.forward(imgs)
        logits = self.head(embs.data)
        logits = F.interpolate(logits, lbls.shape[-2:], mode='bilinear', align_corners=False)
        loss = self.loss_fn(logits, lbls)
        optim = self.optimizers()
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()
        self.lr_scheduler.step()

        log_args = dict(sync_dist=True, rank_zero_only=True)
        self.log('loss/lr', self.lr_scheduler.get_last_lr()[0], **log_args)
        self.log('loss/total', loss.item(), **log_args)

        return loss

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.net.eval()
        self.head.eval()
        self.linear_metrics.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        imgs = batch["img"]
        lbls = batch["label"]
        index = batch["ind"]
        embs = self.forward(imgs)

        logits = self.head(embs)
        preds = F.interpolate(logits, lbls.shape[-2:], mode='bilinear', align_corners=False)
        preds = preds.argmax(1)
        self.linear_metrics.update(preds, lbls)

        return None

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        tb_metrics = self.linear_metrics.compute()

        if self.global_rank == 0:
            print(tb_metrics)

        if self.global_step > 2:
            self.log_dict(tb_metrics, sync_dist=True)

        miou = tb_metrics["mIoU"]
        if self.global_rank == 0 and miou > self.best_miou:
            self.best_miou = miou
            save_dir = self.cfg.basic.eval_save_dir

            if not os.path.isdir('/'.join(save_dir.split('/')[:-1])):
                os.makedirs('/'.join(save_dir.split('/')[:-1]))

            with open(save_dir, 'w') as f:
                for key in tb_metrics.keys():
                    f.write('%s: %.4f\n'%(key, tb_metrics[key]))

        torch.cuda.empty_cache()


    def configure_optimizers(self):
        net_optim = torch.optim.Adam(
            list(self.head.parameters()),
            lr=self.cfg_val.lr
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            net_optim,
            T_max=self.cfg_val.num_finetune_sample // self.cfg_val.ft_batchsize + 5,
            eta_min=1e-4 if self.cfg_val.head == 'fcn' else self.cfg_val.lr
        )

        return [net_optim]


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    ## set random seed
    seed_everything(seed=0)

    ## setup gpus
    gpu_args = dict(devices=cfg.training.num_gpus, accelerator='gpu', strategy='ddp_find_unused_parameters_false')

    ## setup logger
    OmegaConf.set_struct(cfg, False)
    cfg = share_cfg(cfg)

    key = "dataset_seg"
    log_dir = osp.join(cfg.basic.output_root, "logs")
    checkpoint_dir = osp.join(cfg.basic.output_root, "checkpoints")
    prefix= "{}/{}".format(cfg.dataset_val[key].dataset_name, cfg.basic.experiment_name)
    cfg.full_name = prefix
    
    if cfg.basic.resume_from_checkpoint is not None:
        name = osp.dirname(osp.relpath(cfg.basic.resume_from_checkpoint, checkpoint_dir))
    else:
        name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))

    tb_logger = TensorBoardLogger(osp.join(log_dir, name), default_hp_metric=False)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    sys.stdout.flush()

    ## setup datasets
    finetune_dataset = DataWrapper(
        cfg=cfg.dataset_val[key],
        image_set="train",
        label=True,
        pos_images=False,
        pos_labels=False,
        randomaug=False,
        res=cfg.dataset_val[key].ft_res,
        random_crop_res=cfg.dataset_val[key].ft_res
    )
    val_dataset = DataWrapper(
        cfg=cfg.dataset_val[key],
        image_set="val",
        label=True,
        pos_images=False,
    )
    assert cfg.dataset_val[key].n_classes == val_dataset.n_classes

    ## setup model
    model = FinetuneSegmenter(finetune_dataset, val_dataset, cfg)

    trainer = Trainer(
        resume_from_checkpoint=cfg.basic.resume_from_checkpoint,
        check_val_every_n_epoch=cfg.dataset_val[key].check_val_every_n_epoch,
        log_every_n_steps=10,
        logger=tb_logger,
        max_steps=cfg.dataset_val[key].num_finetune_sample // cfg.dataset_val[key].ft_batchsize,
        inference_mode=False,
        precision=cfg.basic.precision,
        enable_progress_bar=cfg.basic.enable_progress_bar,
        callbacks=[
            ModelCheckpoint(
                dirpath=osp.join(checkpoint_dir, name),
                save_top_k=-1,
                every_n_train_steps=10000,
                filename='epoch_{epoch}-step_{step}',
            )
        ],
        **gpu_args
    )

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    prep_args()
    my_app()

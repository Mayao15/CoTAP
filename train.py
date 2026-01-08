from genericpath import exists
import sys
import os
import os.path as osp
import torch.multiprocessing
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend
import torchvision
import pytorch_lightning as pl
import hydra
import random
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from lightning_lite.utilities.seed import seed_everything
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import cosine_scheduler, prep_args, share_cfg, load_checkpoint
from datasets.data_wrapper import DataWrapper
# from datasets.transforms import GaussianBlur


torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32=True


class MaimModel(pl.LightningModule):
    def __init__(self, train_set, val_set, finetune_set, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        # cfg.model.num_classes = n_classes_val
        self.build_model(cfg.model)
        self.build_loss_fn(cfg.loss_fn)
        self.build_transform()
        self.build_tester()
        self.train_set = train_set
        self.val_set = val_set
        self.finetune_set = finetune_set

        self.checked = False
        self.save_hyperparameters(ignore=['train_set', 'val_set', 'finetune_set', 'tester', 'loss_fn'])
        self.ema_schedule = cosine_scheduler(self.cfg.training.ema, 0.9999, self.cfg.training.max_steps + 5, 0, 1)

        # Visualization Config
        base_vis_dir = '/home/czx/temp_datasets/ImageNet2012/val/n01440764/'
        self.vis_img_paths = [
            os.path.join(base_vis_dir, 'ILSVRC2012_val_00000293.JPEG'),
            os.path.join(base_vis_dir, 'ILSVRC2012_val_00002138.JPEG'),
            os.path.join(base_vis_dir, 'ILSVRC2012_val_00003014.JPEG'),
            os.path.join(base_vis_dir, 'ILSVRC2012_val_00006697.JPEG'),
            os.path.join(base_vis_dir, 'ILSVRC2012_val_00007197.JPEG')
        ]
        # Filter out non-existent paths
        self.vis_img_paths = [p for p in self.vis_img_paths if os.path.exists(p)]

    def broadcast(self, x, src):
        return self.all_gather(x)[src]

    def reduce(self, x, mode='sum'):
        if mode == 'sum':
            return self.all_gather(x).sum(0)
        else:
            return self.all_gather(x).mean(0)

    def world_size(self):
        return self.cfg.training.num_gpus

    def build_transform(self):

        def get_transform(jitter_strength, blur_strength):
            color_jitter = torchvision.transforms.ColorJitter(
                0.8 * jitter_strength, 0.8 * jitter_strength, 0.8 * jitter_strength,
                0.2 * jitter_strength
            )
            color_transform = [
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2)
            ]
            blur = torchvision.transforms.GaussianBlur((7,7), sigma=[blur_strength * .1, blur_strength * 2.])
            color_transform.append(torchvision.transforms.RandomApply([blur], p=0.5))
            color_transform = torchvision.transforms.Compose(color_transform)
            return color_transform

        self.color_transform = get_transform(jitter_strength=1, blur_strength=1)

    def build_model(self, cfg):
        dim = cfg.dim
        
        if self.cfg.basic.eval_only:
            from models.model_wrapper import DinoFeaturizer
            self.net = DinoFeaturizer(dim, cfg, require_grad=True, \
                    load_pretrain=False, pretrained_weights=cfg.pretrained_weights)    
            del self.net.model.projection_heads
        elif cfg.arch == "dino":
            from models.model_wrapper import DinoFeaturizer
            self.net = DinoFeaturizer(dim, cfg, require_grad=True, \
                    load_pretrain=True, pretrained_weights=cfg.pretrained_weights)
            self.net_teacher = DinoFeaturizer(dim, cfg, require_grad=False, load_pretrain=False)
            self.net_teacher.ema_update(self.net, 0)
        elif cfg.arch == 'resnet':
            from models.model_wrapper import ResNet
            self.net = ResNet(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.training_model = 0

    def build_loss_fn(self, cfg):
        from losses import LossWrapper
        self.loss_fn = LossWrapper(cfg)

    def build_tester(self):
        from tester.sematic_seg_tester import SematicSegTester
        from tester.img_cls_tester import ImageClsTester

        cfg = self.cfg.dataset_val
        self.tester = []
        for key in cfg.keys():
            if 'seg' in key:
                self.tester.append(SematicSegTester(cfg[key], \
                    'test/%s'%cfg[key].prefix))
            elif 'cls' in key:
                self.tester.append(ImageClsTester(cfg[key], \
                    'test/%s'%cfg[key].prefix))
            else:
                raise NotImplementedError('unknown evaluation type: %s'%key)
        self.tester = torch.nn.ModuleList(self.tester)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            self.cfg.training.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=self.cfg.basic.num_workers, 
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        cfg = self.cfg.dataset_val
        val_loaders = []
        for i, key in enumerate(cfg.keys()):
            # 减少 num_workers 以避免共享内存不足
            num_workers = min(4, self.cfg.basic.num_workers // 2) if hasattr(self.cfg.basic, 'num_workers') else 4
            val_loaders.append(DataLoader(
                self.val_set[i],
                cfg[key].batchsize,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True if num_workers > 0 else False,
            ))
        return val_loaders 

    def test_dataloader(self):
        cfg = self.cfg.dataset_val
        val_loaders = []
        for i, key in enumerate(cfg.keys()):
            # 减少 num_workers 以避免共享内存不足
            num_workers = min(4, self.cfg.basic.num_workers // 2) if hasattr(self.cfg.basic, 'num_workers') else 4
            val_loaders.append(DataLoader(
                self.val_set[i],
                cfg[key].batchsize,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True if num_workers > 0 else False,
            ))
        return val_loaders

    def finetune_dataloader(self):
        cfg = self.cfg.dataset_val
        ft_loaders = []
        for i, key in enumerate(cfg.keys()):
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.finetune_set[i],
                self.cfg.training.num_gpus,
                self.global_rank,
                shuffle=True,
                drop_last=False
            )
            batchsize = cfg[key].ft_batchsize // self.cfg.training.num_gpus
            # 减少 num_workers 以避免共享内存不足，并添加 persistent_workers 减少开销
            num_workers = min(4, self.cfg.basic.num_workers // 2) if hasattr(self.cfg.basic, 'num_workers') else 4
            ft_loaders.append(torch.utils.data.DataLoader(self.finetune_set[i],
                batch_size=batchsize, num_workers=num_workers, sampler=train_sampler,
                persistent_workers=True if num_workers > 0 else False))
        return ft_loaders

    @torch.no_grad()
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net.get_test_features(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        self.net.train()
        self.net_teacher.eval()
        log_args = dict(sync_dist=True, rank_zero_only=True)
        net_optim = self.optimizers()
        nmb_crops = self.cfg.dataset_train.nmb_crops
        head_idx_patch = self.cfg.model.head_idx_patch
        head_idx_cls = self.cfg.model.head_idx_cls

        def merge_input(key, use_transform=False, index=0, subset_index=0):
            # When training with a single dataset (e.g., dataset_train=in1k),
            # `batch` is a list of length 1. Some configs (e.g., coco+in1k)
            # use subset_index=1 for the cls branch; clamp it to avoid IndexError.
            if subset_index >= len(batch):
                subset_index = len(batch) - 1
            if not key in batch[subset_index].keys():
                return None

            if index == 0:
                start_idx, end_idx = 0, nmb_crops[0]
            else:
                start_idx, end_idx = nmb_crops[0], sum(nmb_crops)

            if use_transform:
                tr = self.color_transform
            else:
                tr = lambda x:x

            return torch.cat([
                torch.cat([tr(batch[subset_index][key][i]) for i in range(start_idx, end_idx)]),
                torch.cat([tr(batch[subset_index][key + '_pos'][i]) for i in range(start_idx, end_idx)]),
            ])
        
        def merge_multiple_datasets(key, use_transform=False, index=0):
            if key in batch[head_idx_patch].keys() and batch[head_idx_patch][key] is not None:
                return torch.cat([
                    merge_input(key, use_transform, index, i) for i in range(len(batch))
                ], dim=0)
            else:
                return None

        # Select which dataset subset provides patch/cls branches.
        # For multi-dataset training like coco+in1k, typical setting is patch=0, cls=1.
        # For single-dataset training like in1k, both should fall back to 0.
        subset_idx_patch = getattr(self.cfg.loss_fn.share, 'applied_subset_patch', 0)
        subset_idx_cls = getattr(self.cfg.loss_fn.share, 'applied_subset_cls', 0)
        subset_idx_patch = min(int(subset_idx_patch), len(batch) - 1)
        subset_idx_cls = min(int(subset_idx_cls), len(batch) - 1)

        imgs_patch = merge_input('img', True, 0, subset_idx_patch)
        imgs_cls = merge_input('img', True, 0, subset_idx_cls)
        imgs_tea_patch = merge_input('img', False, 0, subset_idx_patch)
        imgs_tea_cls = merge_input('img', False, 0, subset_idx_cls)
        masks = merge_multiple_datasets('mask')
        bboxes = {k:torch.cat([
            torch.cat([batch[i]["bbox_dict"][k], batch[i]["bbox_dict_pos"][k]]) for i in range(len(batch))
        ], dim=0) for k in batch[0]["bbox_dict"].keys()}

        outputs_tea_patch = self.net_teacher(imgs_tea_patch, return_attn=True, proj_head=head_idx_patch)
        outputs_tea_cls = self.net_teacher(imgs_tea_cls, return_attn=False, proj_head=head_idx_cls)
        outputs_tea = {
            'feat_%d'%head_idx_patch: outputs_tea_patch['feat_%d'%head_idx_patch],
            'cls_%d'%head_idx_cls: outputs_tea_cls['cls_%d'%head_idx_cls],
            'attn': outputs_tea_patch['attn']
        }

        outputs_stu_patch = self.net(imgs_patch, masks, proj_head=head_idx_patch)
        outputs_stu_cls = self.net(imgs_cls, masks, proj_head=head_idx_cls, update_kernel=True)
        outputs_stu = {
            'feat_%d'%head_idx_patch: outputs_stu_patch['feat_%d'%head_idx_patch],
            'cls_%d'%head_idx_cls: outputs_stu_cls['cls_%d'%head_idx_cls]
        }

        if self.cfg.dataset_train.nmb_crops[1] > 0:
            lc_imgs_patch = merge_input('img', True, 1, subset_idx_patch)
            lc_imgs_cls = merge_input('img', True, 1, subset_idx_cls)
            lc_outputs_stu_patch = self.net(lc_imgs_patch, proj_head=head_idx_patch)
            lc_outputs_stu_cls = self.net(lc_imgs_cls, proj_head=head_idx_cls)
            lc_outputs_stu = {
                'feat_%d'%head_idx_patch: lc_outputs_stu_patch['feat_%d'%head_idx_patch],
                'cls_%d'%head_idx_cls: lc_outputs_stu_cls['cls_%d'%head_idx_cls]
            }
        else:
            lc_outputs_stu = None
        lc_outputs_tea = None

        self.net.model.cluster(self)
        kernel = self.net.model.kernel
        samples = self.net.model.cache_samples.samples

        total_loss, losses = self.loss_fn(
            outputs_stu=outputs_stu,
            outputs_tea=outputs_tea,
            masks=masks,
            lc_outputs_stu=lc_outputs_stu,
            lc_outputs_tea=lc_outputs_tea,
            bboxes=bboxes,
            kernel=kernel,
            samples=samples,
            projection_heads=self.net_teacher.model.projection_heads,
            num_subsets=len(batch),
            pl_module=self
        )

        net_optim.zero_grad()
        self.manual_backward(total_loss)

        net_optim.step()

        self.lr_scheduler.step()
        self.net_teacher.ema_update(self.net, self.ema_schedule[self.global_step])

        for i, param_group in enumerate(net_optim.param_groups):
            if i == 0 or i == 2:
                param_group["weight_decay"] = self.wd_schedule[self.global_step]

        self.log('loss/lr', self.lr_scheduler.get_last_lr()[0], **log_args)
        for i in losses:
            if 'loss' in i.keys():
                self.log('loss/%s'%i['name'], i['loss'], **log_args)
        self.log('loss/total', total_loss, **log_args)

        if self.global_step % 10000 == 0 and self.global_step > 0:
            self.print("RESETTING TFEVENT FILE")
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return total_loss

    def finetune(self):
        self.net.eval()
        [t.start(self.global_rank) for t in self.tester]
        self.print('\nStart finetuning...')

        keys = list(self.cfg.dataset_val.keys())
        loaders = self.finetune_dataloader()
        for i in range(len(loaders)):
            epoch = 0
            num_finetune_sample = self.cfg.dataset_val[keys[i]].num_finetune_sample
            loader = loaders[i]
            loader.sampler.set_epoch(epoch)
            loader_iterator = iter(loader)
            iterator = range(0, num_finetune_sample, loader.batch_size * self.cfg.training.num_gpus)
            if self.global_rank == 0 and self.cfg.basic.enable_progress_bar:
                iterator = tqdm(iterator)
            for j in iterator:
                try:
                    batch = next(loader_iterator)
                except:
                    loader.sampler.set_epoch(epoch)
                    epoch += 1
                    loader_iterator = iter(loader)
                    batch = next(loader_iterator)
                batch = batch[0]

                img = batch['img'].cuda()
                label = batch['label'].cuda()
                feats_cls, feats_patch = self.forward(img)

                self.tester[i].finetune_step(
                    feats_cls=feats_cls, 
                    feats_patch=feats_patch,
                    label=label, 
                    pl_module=self
                )

            self.tester[i].end(pl_module=self)

            del loader_iterator
        del loaders
        self.print('\nDone.')

    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        self.net.eval()
        self.finetune()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = batch[0]
        img = batch["img"]
        label = batch["label"]
        feats_cls, feats_patch = self.forward(img)
        keys = list(self.cfg.dataset_val.keys())
        if 'cls' in keys[dataloader_idx]:
            self.tester[dataloader_idx].validation_step(feats_cls, label, self)
        elif 'seg' in keys[dataloader_idx]:
            self.tester[dataloader_idx].validation_step(feats_patch, label, self)
        else:
            raise NotImplementedError

        return None

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        tb_metrics = {}
        for i in range(len(self.tester)):
            tb_metrics.update(self.tester[i].compute(outputs, pl_module=self))

        if self.global_step > 2:
            self.log_dict(tb_metrics, sync_dist=True)

        if self.global_rank == 0 and self.cfg.basic.azureml_logging:
            from azureml.core.run import Run
            run_logger = Run.get_context()
            for metric, value in tb_metrics.items():
                run_logger.log(metric, value)

        # Online Visualization (Rank 0 only)
        if self.global_rank == 0:
            self.run_visualization()

        torch.cuda.empty_cache()

    def run_visualization(self):
        if not self.vis_img_paths:
            return

        try:
            # Imports here to avoid circular dependencies
            from visualize import visualize_tsne, visualize_similarity, visualize_entropy, preprocess_image
            from einops import rearrange
            import numpy as np
            
            # Construct save path
            exp_name = self.cfg.full_name.replace('/', '_')
            save_dir = os.path.join(self.cfg.basic.output_root, "vis_results", exp_name, f"epoch_{self.current_epoch}_step_{self.global_step}")
            os.makedirs(save_dir, exist_ok=True)
            
            self.net.eval()
            
            total_vis_entropy = 0.0
            num_vis_images = 0
            
            for img_path in self.vis_img_paths:
                try:
                    img_tensor, img_pil = preprocess_image(img_path, 480, device=self.device)
                    
                    # Use underlying model to avoid DDP synchronization deadlocks since we only run on Rank 0
                    net_to_use = self.net
                    if hasattr(self.net, 'module'):
                        net_to_use = self.net.module

                    with torch.no_grad():
                        _, feats_patch = net_to_use.get_test_features(img_tensor)
                        
                    # Reshape logic (copied from visualize.py main)
                    h, w = 0, 0
                    if feats_patch.ndim == 4:
                        features = rearrange(feats_patch, 'b d h w -> (b h w) d')
                        h, w = feats_patch.shape[2], feats_patch.shape[3]
                    elif feats_patch.ndim == 3:
                        features = rearrange(feats_patch, 'b n d -> (b n) d')
                        n = features.shape[0]
                        h = w = int(np.sqrt(n))
                        
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    output_prefix = os.path.join(save_dir, base_name)
                    
                    # Run visualizations
                    # n_clusters default 4
                    visualize_tsne(features, img_pil, h, w, output_prefix, n_clusters=4)
                    visualize_similarity(features, img_pil, h, w, output_prefix)
                    mean_ent = visualize_entropy(features, output_prefix)
                    
                    if mean_ent is not None:
                        total_vis_entropy += mean_ent
                        num_vis_images += 1
                    
                except Exception as e:
                    self.print(f"Error visualizing {img_path}: {e}")
                    continue

            if num_vis_images > 0:
                avg_vis_entropy = total_vis_entropy / num_vis_images
                # Fix deadlock: Do NOT use sync_dist=True here because only Rank 0 executes this code.
                # Other ranks do not call this log, causing Rank 0 to wait indefinitely for sync.
                self.log('vis/avg_entropy', avg_vis_entropy, rank_zero_only=True)
                self.print(f"Average visualization entropy: {avg_vis_entropy:.4f}")

            self.print(f"Visualization saved to {save_dir}")
            
        except Exception as e:
            self.print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

    def on_test_epoch_start(self):
        load_checkpoint(self.net, self.cfg.basic.resume_test, self.cfg.model.vit_type)
        self.net.eval()
        self.finetune()
        self.tester.cuda()

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.net.eval()
        batch = batch[0]
        img = batch["img"]
        label = batch["label"]
        feats_cls, feats_patch = self.forward(img)
        keys = list(self.cfg.dataset_val.keys())
        if 'cls' in keys[dataloader_idx]:
            self.tester[dataloader_idx].validation_step(feats_cls, label, self)
        elif 'seg' in keys[dataloader_idx]:
            self.tester[dataloader_idx].validation_step(feats_patch, label, self)
        else:
            raise NotImplementedError

        return None

    def test_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        tb_metrics = {}
        for i in range(len(self.tester)):
            tb_metrics.update(self.tester[i].compute(outputs, pl_module=self))

        if self.global_rank == 0:
            print(tb_metrics)
            if self.cfg.basic.eval_save_dir is None:
                weight_name = self.cfg.basic.resume_test
                save_dir = os.path.join(
                    'eval_results',
                    list(self.cfg.dataset_val.keys())[0],
                    self.cfg.model.model_type + '_' + weight_name.split('/')[-1] + '.txt'
                )
            else:
                save_dir = self.cfg.basic.eval_save_dir

            if not os.path.isdir('/'.join(save_dir.split('/')[:-1])):
                os.makedirs('/'.join(save_dir.split('/')[:-1]))
            with open(save_dir, 'w') as f:
                for key in tb_metrics.keys():
                    f.write('%s: %.4f\n'%(key, tb_metrics[key]))

    def configure_optimizers(self):

        def get_params(model):

            head_params_named = []
            backbone_params_named = []
            for name, param in model.named_parameters():
                if name.startswith("model.projection_heads"):
                    head_params_named.append((name, param))
                else:
                    backbone_params_named.append((name, param))

            backbone_params = self.exclude_from_wt_decay(
                backbone_params_named,
                weight_decay=self.cfg.training.weight_decay,
                lr=self.cfg.training.lr
            )
            head_params = self.exclude_from_wt_decay(
                head_params_named,
                weight_decay=self.cfg.training.weight_decay,
                lr=self.cfg.training.lr * 10
            )
            params = backbone_params + head_params

            assert len(backbone_params_named) > 0 and len(head_params_named) > 0

            net_optim = torch.optim.AdamW(
                params,
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
                foreach=True
            )

            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                net_optim,
                T_max=self.cfg.training.max_steps + 5,
                eta_min=3e-6
            )

            self.wd_schedule = cosine_scheduler(
                self.cfg.training.weight_decay,
                self.cfg.training.weight_decay_end,
                self.cfg.training.max_steps + 5
            )

            return net_optim

        return [get_params(self.net)]

    @staticmethod
    def exclude_from_wt_decay(named_params, weight_decay: float, lr: float):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            # do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                excluded_params.append(param)
            else:
                params.append(param)
        return [{'params': params, 'weight_decay': weight_decay, 'lr': lr},
                {'params': excluded_params, 'weight_decay': 0., 'lr': lr}]

@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    ## set random seed
    seed_everything(seed=0)

    ## setup gpus
    gpu_args = dict(devices=cfg.training.num_gpus, accelerator='gpu', strategy=DDPStrategy(find_unused_parameters=True))

    ## setup logger
    OmegaConf.set_struct(cfg, False)
    cfg = share_cfg(cfg)

    log_dir = osp.join(cfg.basic.output_root, "logs")
    checkpoint_dir = osp.join(cfg.basic.output_root, "checkpoints")
    prefix= "{}/{}".format(cfg.dataset_train.dataset_name, cfg.basic.experiment_name)
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
    train_dataset = DataWrapper(
        cfg=cfg.dataset_train,
        image_set="train",
        label=False,
        pos_images=not cfg.basic.eval_only,
        pos_labels=False,
        randomaug=True,
        use_leopart_transform=True,
        num_neighbors_chosen=cfg.training.num_neighbors_chosen,
        use_mask=cfg.training.use_mask
    )

    val_datasets, finetune_datasets = [], []
    for key in cfg.dataset_val.keys():
        val_datasets.append(DataWrapper(
            cfg=cfg.dataset_val[key],
            image_set="val",
            label=True,
            pos_images=False,
        ))
        # training set for learning linear mapping
        finetune_datasets.append(DataWrapper(
            cfg=cfg.dataset_val[key],
            image_set="train",
            label=True,
            pos_images=False,
            pos_labels=False,
            randomaug=False,
            res=cfg.dataset_val[key].ft_res,
        ))
        cfg.dataset_val[key].num_classes = val_datasets[-1].n_classes

    ## setup model
    model = MaimModel(train_dataset, val_datasets, finetune_datasets, cfg)

    trainer = Trainer(
        resume_from_checkpoint=cfg.basic.resume_from_checkpoint,
        val_check_interval=cfg.basic.val_freq,
        log_every_n_steps=cfg.basic.scalar_log_freq,
        logger=tb_logger,
        # Avoid misleading metrics during Lightning "sanity checking", which by default
        # runs only a couple of val batches and can produce artificially high/unstable KNN.
        num_sanity_val_steps=0,
        max_steps=1 if cfg.basic.eval_only else cfg.training.max_steps,
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

    if cfg.basic.eval_only:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == "__main__":
    prep_args()
    my_app()

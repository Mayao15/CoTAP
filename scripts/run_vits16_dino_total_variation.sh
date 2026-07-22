#!/bin/bash

torchrun --nproc_per_node=8 train.py \
loss_fn=full_loss training.num_gpus=8 dataset_train=coco+in1k training.batch_size=32 training.use_mask=False training.max_steps=240000 training.lr=3e-5 model.n_projection_head=2 model.nmb_prototypes=[512,4096] \
basic.experiment_name="vits16/dino+total_variation" \
model.vit_type=depth \
model.feature_regularization=total_variation \
model.dim=768 \
training.ema=0.9997 \
loss_fn.spatial_regularization.weight=0.01 \
loss_fn.innersample.weight_patch_inner=1 \
loss_fn.innersample.weight_cls_inner=1 \
loss_fn.intrasample.weight_patch_intra=1 \
loss_fn.intrasample.weight_cls_intra=1

wait

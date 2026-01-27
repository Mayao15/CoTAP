#!/bin/bash

torchrun --nproc_per_node=8 train.py \
loss_fn=full_loss training.num_gpus=8 dataset_train=coco+in1k training.batch_size=32 training.use_mask=False training.max_steps=240000 training.lr=3e-5 model.n_projection_head=2 model.nmb_prototypes=[512,4096] \
basic.experiment_name="vits16/dino+cotap" \
model.vit_type='depth' \
model.selective_layer_start=11 \
model.selective_cache_num=64 \
model.selective_kernel_size=3 \
model.dim=768 \
training.ema=0.9997 \
loss_fn.innersample.weight_patch_inner=1 \
loss_fn.innersample.weight_cls_inner=1 \
loss_fn.intrasample.weight_patch_intra=1 \
loss_fn.intrasample.weight_cls_intra=1

wait

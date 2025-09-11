#!/bin/bash

devices=(0,1,2,3,4,5,6,7)

model=('ours/vits16/dino+cotap' \
'ours/vits16/ibot+cotap' \
'ours/vits16/leopart+cotap' \
'ours/vits16/mugs+cotap'
)

save_name=('dino+cotap' \
'ibot+cotap' \
'leopart+cotap' \
'mugs+cotap'
)

n_devices=${#devices[@]}
for((i=0;i<${#model[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${devices[$[i%n_devices]]} python train.py \
    basic.eval_only=True model=vit_small_16 training.num_gpus=8 \
    basic.experiment_name=in1k_knn_${save_name[$i]} \
    basic.resume_test=./checkpoints/${model[$i]}.ckpt \
    basic.eval_save_dir=./eval_results_cls/competitor/${model[$i]} \
    dataset_val=imagenet1k
    if [ $[(i+1)%n_devices] -eq 0 ]
    then
        wait
    fi
done
wait

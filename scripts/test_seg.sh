#!/bin/bash

devices=(0,1,2,3,4,5,6,7)
evaluate_group="cocostuff27_linear" # voc_linear voc_fcn cocostuff27_linear cocostuff27_fcn cityscapes_linear cityscapes_fcn ade20k_linear ade20k_fcn 

lrs=('3e-3')

model=('cotap/dino+cotap'
)

save_name=('cotap' \
)

n_devices=${#devices[@]}

for((k=0;k<${#lrs[@]};k++))
do
    for((i=0;i<${#model[@]};i++))
    do
        CUDA_VISIBLE_DEVICES=${devices[$[i%n_devices]]} python train_segmentation.py \
        basic.eval_only=False model=vit_small_16 training.num_gpus=8 \
        basic.experiment_name=${evaluate_group}_${lrs[$k]}_${save_name[$i]} \
        basic.resume_test=./checkpoints/${model[$i]}.ckpt \
        basic.eval_save_dir=./eval_results_seg/${evaluate_group}/lr_${lrs[$k]}/${model[$i]} \
        dataset_val.dataset_seg.lr=${lrs[$k]} \
        dataset_val=${evaluate_group}
        if [ $[(i+1)%n_devices] -eq 0 ]
        then
            wait
        fi
    done
    wait
done

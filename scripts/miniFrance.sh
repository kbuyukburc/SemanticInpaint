#!/bin/bash

# Training
export OPENAI_LOGDIR='OUTPUT/MiniFrance-SDM-256CH'
python image_train.py --data_dir ../output2/balanced_patches2 --dataset_mode miniFrance --lr 1e-4 \
    --batch_size 4 \
    --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 128 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
    --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
    --use_checkpoint True --num_classes 16 \
    --class_cond True --no_instance True
#!/bin/bash

# Inpainting Finetune
export OPENAI_LOGDIR='OUTPUT/MiniFrance-SDM-256CH-INPAINT'
python image_train.py --data_dir ../output2/balanced_patches2 --dataset_mode miniFrance --lr 2e-5 \
    --batch_size 1 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 \
    --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
    --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
    --use_checkpoint True --num_classes 16 \
    --class_cond True --no_instance True \
    --resume_checkpoint OUTPUT/MiniFrance-SDM-256CH/model_best.pt \
    --inpainting True

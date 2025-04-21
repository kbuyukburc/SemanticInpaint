#!/bin/bash

# Testing
export OPENAI_LOGDIR='OUTPUT/MiniFrance-SDM-256CH-TEST-INPAINT'
python image_sample.py --data_dir ../output2/balanced_patches2 --dataset_mode miniFrance \
    --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
    --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 16 \
    --class_cond True --no_instance True --batch_size 1 --num_samples 6 \
    --model_path OUTPUT/MiniFrance-SDM-256CH-INPAINT/model016000.pt \
    --results_path RESULTS/MiniFrance-SDM-256CH-INPAINT --s 1.5 \
    --inpainting True

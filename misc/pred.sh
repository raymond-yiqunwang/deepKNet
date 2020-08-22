#!/bin/bash

jobname=test1
python predict.py \
    --root ./data_gen/data_MIC_C125/ \
    --modelpath ./hyper_params/MIC_runs/checkpoints_history/MIC_run3_model_best.pth.tar \
    --target MIC \
    --nclass 2 \
    --threshold 0.9 \
    --run_name ${jobname} \
    --gpu_id 0 \
    --npoint 125 \
    --point_dim 4 \
    --padding zero \
    --data_aug True \
    --rot_all True \
    --permutation True \
    --conv_dims 4 128 512 \
    --nbert 4 \
    --fc_dims 512 256 128 \
    --pool CLS \
    --batch_size 256 \
    --print_freq 1 \
    | tee 2>&1 ${jobname}.log


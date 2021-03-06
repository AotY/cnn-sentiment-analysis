#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.

export CUDA_VISIBLE_DEVICES=5,6,7

python train.py \
    --data_dir data/rt-polaritydata \
    --embedding_size 300 \
    --dropout_p 0.8 \
    --num_classes 2 \
    --max_norm 50.0 \
    --max_len 50 \
    --min_count 3 \
    --lr 0.005 \
    --epochs 15 \
    --start_epoch 1 \
    --batch_size 64 \
    --eval_split 0.1 \
    --test_split 0.2 \
    --seed 7 \
    --device cuda:0 \
    --log_interval 20 \
    --log_file ./logs/train.log \
    --model_save_path ./models \
    --train_or_eval train \
    --checkpoint ./models/checkpoint.epoch-5.pth \

/




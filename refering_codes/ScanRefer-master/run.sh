#!/bin/bash

module load cuda/11.1.0
TORCH_DISTRIBUTED_DEBUG=INFO /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u scripts/train.py \
 --anchors "cot" --cot_type "cross" \
 --predict_lang_anchors True --max_num_anchors 7 --train_data_percent 0.1 \
 --use_multiview --use_normal --batch_size 60 --val_step 500000 \
 --verbose 100 \
 --newdataset True \
 --feedcotpath True \
 --cot_trans_lr_scale 0 \
 --gpu 0
#!/bin/bash
source ../hyperparameters.txt

CUDA_VISIBLE_DEVICES=0 python3 $lm_dir/main.py \
       --lm_data $lm_data_dir \
       --ccg_data $ccg_data_dir \
       --cuda \
       --epochs $epochs \
       --model $model \
       --nhid $num_hid \
       --save $model_dir/lstm_multi.pt \
       --save_lm_data $model_dir/lstm_multi.bin \
       --log-interval $log_freq \
       --batch $batch_size \
       --dropout $dropout \
       --lr $lr \
       --trainfname $train \
       --validfname $valid \
       --testfname $test

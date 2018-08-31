#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_FN.py --dataset_option=normal --path_opt options/models/VG-MSDN.yaml --rpn output/RPN.h5

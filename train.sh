#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_FN.py --path_opt=options/FN_v4/map_v1.yaml --rpn=output/RPN/RPN_region.h5

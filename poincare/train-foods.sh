#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python3 embed.py \
       -dim 3 \
       -lr 0.3 \
       -epochs 300 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -manifold poincare \
       -dset food/food_closure.csv \
       -checkpoint checkpoint/foods.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 2

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import argparse
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/media/smartcoop/99/5-1/test/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.exp_name = 'yolox_s'
        
        self.enable_mixup = True
        self.mosaic_scale = (0.1, 2)
        self.degrees = 10.0

        self.num_classes = 600

        self.max_epoch = 50
        self.data_num_workers = 2
        self.eval_interval = 10
        self.no_aug_epochs = 15

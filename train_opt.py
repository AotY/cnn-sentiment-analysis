#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.


import argparse

def data_set_opt(parser):
    # Data set options
    group = parser.add_argument_group('data set opt')
    group.add_argument('--data_dir',
                       type=str,
                       required=True,
                       help='path to the data dir.')

    group.add_argument('--min_count',
                       type=int,
                       required=True,
                       hlep='threshold for word frequency.')

def model_opt(parser):
    group = parser.add_argument_group('cnn model opt')

    group.add_argument('--embedding_size',
                       type=int,
                       default=100,
                       help='embedding size.')

    group.add_argument('--num_classes',
                       type=int,
                       default=2,
                       help='num classes.')

    group.add_argument('--dropout_p',
                       type=float,
                       default=0.5,
                       help='dropout ratio.')


def train_opt(parser):
    group = parser.add_argument_group('train opt')

    group.add_argument('--lr', type=float, default=0.001,
                       help='initial learning rate')

    group.add_argument('--epochs', type=int, default=5,
                       help='upper epoch limit')

    group.add_argument('--start_epoch',
                       type=int,
                       default=1,
                       help='start epoch')

    group.add_argument('--seed',
                       type=int,
                       default=7,
                       help='random seed')

    group.add_argument('--device',
                       type=str,
                       default='cuda',
                       help='use cuda or cpu.')

    group.add_argument('--log_interval',
                       type=int,
                       default=100,
                       help='report interval')

    group.add_argument('--model_save_path',
                       type=str,
                       default='./models',
                       help='path to save models')

    group.add_argument('--log_file',
                       type=str,
                       help='path to save logger.')

    group.add_argument('--batch_size',
                       type=int,
                       default=128,
                       help='batch size')

    group.add_argument('--eval_split',
                       type=float,
                       help='split datas for eval.')

    group.add_argument('--test_split',
                       type=float,
                       help='split datas for test.')

    group.add_argument('--train_or_eval',
                       type=str,
                       help='select train model or eval model')

    group.add_argument('--checkpoint',
                       type=str, help='path to model s checkpoint.')


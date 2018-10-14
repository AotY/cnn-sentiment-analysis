#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Train CNN Classification.
"""

import os
import sys

import math
import argparse
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from model import CSC
from data_set import DataSet
from train_opt import data_set_opt, model_opt, train_opt


program = os.path.basename(sys.argv[0])

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger(program)
logger.info("Running %s", ' '.join(sys.argv))

# get optional parameters
parser = argparse.ArgumentParser(description=program,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
data_set_opt(parser)
model_opt(parser)
train_opt(parser)
opt = parser.parse_args()

device = torch.device(opt.device)
logger.info('device: {}'.format(device))


def train_epochs(model, data_set, optimizer, criterion):
    model.train()  # set to train state
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logger.info('---------------- epoch: %d --------------------' % (epoch))
        data_set.reset_data('train')

        total_loss = 0

        iters = math.ceil(data_set.size_dict['train'] / opt.batch_size )

        for iter in range(1, 1 + iters):
            inputs, inputs_length, labels = data_set.next_batch('train', opt.batch_size)

            loss = train(inputs, inputs_length, labels,
                         model, optimizer, criterion)

            total_loss += loss

            if iter % opt.log_interval == 0:
                avg_loss = total_loss / opt.log_interval
                # reset total_loss
                total_loss = 0
                logger.info('train epoch: %d\titer/iters: %d%%\tloss: %.4f' %
                            (epoch, iter / iters * 100, avg_loss))

        # eval
        eval_loss, eval_accuracy, eval_recall, eval_f1 = evaluate(model, data_set, optimizer, criterion)
        # save model of each epoch
        save_state={
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': eval_loss,
            'accuracy': eval_accuracy,
            'recall': eval_recall,
            'eval_f1': f1
        }

        # optimizer.state_dict()
        save_checkpoint(state=save_state,
                        is_best = False,
                        filename = os.path.join(opt.model_save_path, 'checkpoint.%s.epoch-%d.pth' % (opt.decoder_attn_type, epoch)))



def train(inputs, inputs_length, labels, model, optimizer, criterion):

    output = model(inputs, opt.batch_size)

    loss = 0

    # clear the gradients off all optimzed
    optimizer.zero_grad()

    loss = criterion(output, labels)

    # computes the gradient of current tensor, graph leaves.
    loss.backward()

    # Clip gradients: gradients are modified in place
    #  _ = nn.utils.clip_grad_norm_(model.parameters(), opt.max_norm)

    # performs a single optimization setp.
    optimizer.step()

    return loss.item()


def evaluate(model, data_set, criterion, task='eval'):
    model.eval()  # set to evaluate state
    with torch.no_grad():
        data_set.reset_data(task)

        total_loss = 0

        total_accuracy = 0
        total_recall = 0
        total_f1 = 0

        iters = math.ceil(data_set.size_dict[task] / opt.batch_size )

        for iter in range(1, 1 + iters):
            inputs, inputs_length, labels = data_set.next_batch(task, opt.batch_size)

            output = model(inputs, inputs_length)

            loss = criterion(output, labels)

            total_loss += loss.item()

            # accuracy


    return [item / iters for item in [total_loss, total_accuracy, total_recall, total_f1]]

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    '''
    Saving a model in pytorch.
    :param state: is a dict object, including epoch, optimizer, model etc.
    :param is_best: whether is the best model.
    :param filename: save filename.
    :return:
    '''
    save_path = os.path.join(opt.model_save_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth')


def load_checkpoint(filename='checkpoint.pth'):
    logger.info("Loding checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    return checkpoint

def build_model(vocab):
    model = CSC(vocab_size=vocab.size,
                embedding_size=opt.embedding_size,
                dropout_p=opt.dropout_p,
                padding_idx=vocab.pad_id,
                num_classes=opt.num_classes)

    print(model)
    model.to(device=device)

    return model


def build_optimizer(model):
    optimizer = optim.Adam(model.parameters(), opt.lr)
    return optimizer

def build_criterion():
    criterion = nn.NLLLoss(reduction='elementwise_mean')
    return criterion





if __name__ == "__main__":

    # train()
    data_set = DataSet(opt.data_dir, opt.max_len,
                       opt.min_count, opt.eval_split,
                       opt.test_split, device)

    model = build_model(data_set.vocab)

    # optim
    optimizer = build_optimizer(model)

    # loss function
    criterion = build_criterion()

    # Loading checkpoint
    checkpoint = None
    if opt.checkpoint:
        checkpoint = load_checkpoint(opt.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        recall = checkpoint['recall']
        f1 = checkpoint['f1']
        logger.info('checkpoint loss: {}'.format(loss))
        logger.info('checkpoint accuracy: {}'.format(accuracy))
        logger.info('checkpoint recall: {}'.format(recall))
        logger.info('checkpoint f1: {}'.format(f1))

    if opt.train_or_eval == 'test':
        evaluate(model, data_set, criterion, task='test')
    elif opt.train_or_eval == 'train':
        # train
        train_epochs(model, data_set, optimizer, criterion)
    elif opt.train_or_eval == 'eval':
        evaluate(model, data_set, criterion, task='eval')




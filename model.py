#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
CNN For Sentiment Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSC(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 dropout_p=0.0,
                 padding_idx=0,
                 num_classes=2):
        super(CSC, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.dropout = nn.Dropout(dropout_p)

        # conv2d
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(3, embedding_size),
                               stride=1) # output: [batch_size, output_channels, max_len - 3 + 1, 1]
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(3, 256),
                               stride=1) # output: [batch_size, output_channels, , 1]

        # linear
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, num_classes)

        # log soft max
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, batch_size):
        """
        inputs: [batch_size, len]
        """
        embedded = self.embedding(inputs) # [batch_size, len, embedding_size]
        embedded = self.dropout(embedded)

        # conv
        conv1_output = self.conv1(embedded.unsqueeze(1))
        conv1_output = F.relu(conv1_output)
        conv1_output = conv1_output.squeeze(3).transpose(1, 2) # [batch_size, len - stride + 1, output_channels]
        print('conv1_output shape: {}'.format(conv1_output.shape))

        conv2_output = self.conv2(conv1_output.unsqueeze(1))
        conv2_output = F.relu(conv2_output)
        print('conv2_output shape: {}'.format(conv2_output.shape))

        # max pool [batch_size, output_channels, 1, 1]
        max_pool_output = F.max_pool2d(conv2_output, kernel_size=(conv2_output.shape[2], 1))
        print('max_pool_output shape: {}'.format(max_pool_output.shape))

        # [batch_size, output_channels] out_channels
        fc1_input = max_pool_output.view(batch_size, -1)
        print('fc1_input shape: {}'.format(fc1_input.shape))

        fc1_output = self.fc1(fc1_input)
        fc2_output = self.fc2(fc1_output)

        # softmax
        output = self.softmax(fc2_output)

        return output



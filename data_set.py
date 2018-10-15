#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import os
import random

import torch

from vocab import Vocab
from nltk.tokenize import word_tokenize

class DataSet:
    def __init__(self,
            data_dir,
            max_len,
            min_count,
            eval_split=0.1,
            test_split=0.2,
            device=None):

        self.max_len = max_len
        self.min_count = min_count
        self.device = device

        self.vocab = Vocab()
        self._data_dict = None
        self._indicator_dict = None

        self.load_data(data_dir, eval_split, test_split)

    def load_data(self, data_dir, eval_split, test_split):
        datas = []
        filenames = os.listdir(data_dir)
        for filename in filenames:
            file_path = os.path.join(data_dir, filename)
            if filename.endswith('pos'):
                label = 1
            elif filename.endswith('neg'):
                label = 0
            else:
                continue

            # token
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = [word.lower() for word in word_tokenize(line.rstrip(), language='english')]
                    self.vocab.add_words(tokens)

                    datas.append((tokens, label))

        self.vocab.build_vocab(self.min_count)

        random.shuffle(datas)

        n_train = int((1 - eval_split - test_split) * len(datas))
        n_eval = int(eval_split * len(datas))
        n_test = len(datas) - n_train - n_eval

        self._data_dict = {
                'train': datas[:n_train],
                'eval': datas[n_train: n_train + n_eval],
                'test': datas[n_train + n_eval: ]
                }

        self._indicator_dict = {
                'train': 0,
                'eval': 0,
                'test': 0
                }
        self.size_dict = {
                'train': n_train,
                'eval': n_eval,
                'test': n_test
                }

    def reset_data(self, task):
        random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def load_all(self, task):
        return self._data_dict[task]

    def next_batch(self, task, batch_size):
        next_indicator = self._indicator_dict[task] + batch_size
        if next_indicator > self.size_dict[task]:
            self.reset_data(task)
            next_indicator = batch_size

        inputs = torch.ones((batch_size, self.max_len),
                dtype=torch.long,
                device=self.device) * self.vocab.pad_id
        inputs_length = []
        labels = torch.zeros(batch_size,
                dtype=torch.long,
                device=self.device)
        bacth_datas = self._data_dict[task][self._indicator_dict[task]: next_indicator]
        for batch_id, (tokens, label) in enumerate(bacth_datas):
            tokens_id = self.vocab.words_to_id(tokens)
            tokens_id = tokens_id[-min(self.max_len, len(tokens_id)): ]
            inputs_length.append(len(tokens_id))

            for i, tid in enumerate(tokens_id):
                inputs[batch_id, i] = tid

            labels[batch_id] = label

        self._indicator_dict[task] = next_indicator

        return inputs, inputs_length, labels

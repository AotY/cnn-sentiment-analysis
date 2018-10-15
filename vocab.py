#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Vocab
"""


PAD_id = 0
UNK_id = 1

PAD = 'PAD'
UNK = 'UNK'

class Vocab:
    def __init__(self):
        self.word2idx = {'PAD': 0, 'UNK': 1}
        self.word2count = {}
        self.idx2word = {}
        self.n_words = 2

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    ''' filtering words by min count'''
    def build_vocab(self, min_count=3):
        sorted_list = sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)
        sorted_list = [item for item in sorted_list if item[1] > min_count]
        for word, _ in sorted_list:
            self.word2idx[word] = self.n_words
            self.n_words += 1

        # init idx2word
        self.idx2word = {v: k for k, v in self.word2idx.items()}


    def id_to_word(self, id):
        return self.idx2word.get(id, UNK)

    def ids_to_word(self, ids):
        words = [self.id_to_word(id) for id in ids]
        return words

    def word_to_id(self, word):
        return self.word2idx.get(word, UNK_id)

    def words_to_id(self, words):
        word_ids = [self.word_to_id(word) for word in words]
        return word_ids

    @property
    def size(self):
        return len(self.word2idx)

    @property
    def pad_id(self):
        return PAD_id

    @property
    def unk_id(self):
        return UNK_id




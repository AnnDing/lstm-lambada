#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range


import numpy as np
import random
import Constants


class minibatch_loader:
    def __init__(self, data, batch_size, shuffle=True, sample=1.0, punc=True):
        self.batch_size = batch_size

        self.doc = data['src']
        self.ans = data['tgt']
        self.punc = punc

        if sample == 1.0:
            self.data = data
        else:
            # TODO
            '''
            self.data = random.sample(
                data, int(sample * len(data)))
            '''
        
        # self.max_num_cand = max(list(map(lambda x: len(x[3]), self.questions)))
        # self.max_word_len = MAX_WORD_LEN

        self.shuffle = shuffle
        self.reset()

    def __len__(self):
        return len(self.batch_pool)

    def __iter__(self):
        """make the object iterable"""
        return self

    def reset(self):
        """new iteration"""
        self.ptr = 0

        self.batch_pool = []
        n = len(self.doc)
        idx_list = np.arange(0, n, self.batch_size)

        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)

            self.doc = np.array(self.doc)[idx]
            self.ans = np.array(self.ans)[idx]

        for idx in idx_list:
            self.batch_pool.append(
                np.arange(idx, min(idx + self.batch_size, n)))

    def __next__(self):
        """load the next batch"""
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        ixs = self.batch_pool[self.ptr]

        # maximum number of vocabulary appears per sentence
        # TODO: right now candidate answer is selected from only document
        max_cand_num = max(len(set(self.doc[idx])) for idx in ixs)

        doc_len = [len(self.doc[idx]) for idx in ixs]

        curr_max_doc_len = np.max(doc_len)
        curr_batch_size = len(ixs)

        # document
        docs = np.zeros(
            (curr_batch_size, curr_max_doc_len), dtype='int32')
        # correct answer
        anss = np.zeros((curr_batch_size, ), dtype='int32')

        # document word mask
        docs_mask = np.zeros(
            (curr_batch_size, curr_max_doc_len), dtype='int32')

        # candidate answers lookup table
        cand_position = np.zeros(
            (curr_batch_size, max_cand_num, curr_max_doc_len), dtype='float32')

        for n, ix in enumerate(ixs):

            # document, query and candidates
            docs[n, : len(self.doc[ix])] = np.array(self.doc[ix])

            docs_mask[n, : len(self.doc[ix])] = 1

            if self.punc == True:
                cands = []
                for doc_word in self.doc[ix]:
                    if doc_word < 2 or doc_word > 39:
                        if doc_word not in cands:
                            cands += [doc_word]
            else:
                cands = list(set(self.doc[ix]))

            cands += [Constants.ANS_NO]

            anss[n] = len(cands) - 1
            # search candidates in doc
            for it, cc in enumerate(cands):
                index = [ii for ii in range(len(self.doc[ix])) if self.doc[ix][ii] == cc]
                cand_position[n, it, index] = 1

                if int(self.ans[ix][0]) == int(cc):
                    anss[n] = it  # answer

        self.ptr += 1

        return docs, anss, docs_mask, cand_position

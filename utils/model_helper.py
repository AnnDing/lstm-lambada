#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import torch
import Constants
import torch.nn.functional as F


def tcat(t1, t2):
    return torch.cat([t1, t2], axis=2)


def tsum(t1, t2):
    return t1 + t2


def tmul(t1, t2):
    return torch.mul(t1, t2)



def calculate_attention(doc, qry):
    # doc: B x N x D
    # qry: B x D x 1
    # inter: B x N x Q
    # mask (qry): B x Q

    # score: B x N x 1
    score = torch.bmm(doc, qry).view(doc.size(0), -1)
    attn_mask = score.data.eq(Constants.PAD)
    score.data.masked_fill_(attn_mask, -float('inf'))
    attention = F.softmax(score, dim = 1)

    return attention


def attention_sum(attention, cand_position): #, cand, cloze, cand_mask):
    # B C_N N * B N 1
    pred = torch.bmm(cand_position, attention).squeeze(-1)
    return pred


def gru(inputs, mask, cell):
    """
    Args:
    inputs: batch_size x seq_len x n_feat
    mask: batch_size x seq_len
    cell: GRU/LSTM/RNN
    """

    seq_lengths = torch.sum(mask, dim=-1).squeeze(-1)

    sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
    
    index_sorted_idx = sorted_idx\
        .view(-1, 1, 1).expand_as(inputs)

    sorted_inputs = inputs.gather(0, index_sorted_idx.long())

    packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
        sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)

    out, _ = cell(packed_seq)

    unpacked, unpacked_len = \
        torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True)
    _, original_idx = sorted_idx.sort(0, descending=False)
    unsorted_idx = original_idx\
        .view(-1, 1, 1).expand_as(unpacked)
    output_seq = unpacked.gather(0, unsorted_idx.long())

    return output_seq, seq_lengths


def crossentropy(pred, target):
    """
    pred: B x C
    target: B
    """
    idx = target.unsqueeze(1)

    logit = pred.gather(1, idx)

    return - torch.log(logit)

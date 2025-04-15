from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

"""
This module contains utility functions for building CMN and R2Gen models.
It also includes utility functions for handling packed sequences and attention features for AttBase, CMN and R2Gen model.

Functions:
    clones(module, N): Creates N identical copies of a given module.
    subsequent_mask(size): Generates a mask to prevent attention to subsequent positions in a sequence.
    attention(query, key, value, mask=None, dropout=None): Computes scaled dot-product attention.
    memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32): Implements memory querying and responding with top-k selection.

    sort_pack_padded_sequence(input, lengths):
        Sorts and packs a padded sequence for RNN processing.
        Args:
            input (Tensor): Input tensor of shape (batch_size, seq_len, ...).
            lengths (Tensor): Lengths of each sequence in the batch.
        Returns:
            PackedSequence: Packed sequence for RNN processing.
            Tensor: Indices to restore the original order.

    pad_unsort_packed_sequence(input, inv_ix):
        Pads and unsorts a packed sequence back to its original order.
        Args:
            input (PackedSequence): Packed sequence to be unsorted.
            inv_ix (Tensor): Indices to restore the original order.
        Returns:
            Tensor: Padded sequence in the original order.

    pack_wrapper(module, att_feats, att_masks):
        Wraps a module to handle packed sequences for attention features.
        Args:
            module (nn.Module): The module to apply to the packed sequence.
            att_feats (Tensor): Attention features.
            att_masks (Tensor): Attention masks.
        Returns:
            Tensor: Output of the module applied to the attention features.
"""

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)
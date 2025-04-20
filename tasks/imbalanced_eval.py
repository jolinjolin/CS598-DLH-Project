import json
import numpy as np
import re

"""
This module provides functions for preprocessing text data and evaluating imbalanced datasets.

Functions:
----------
1. preprogress(f: str) -> list:
    Reads a file line by line, extracts and processes lines starting with the keyword 'Impression:'.
    Returns a list of processed lines.

    Parameters:
    - f (str): The file path to read and process.

    Returns:
    - list: A list of processed lines containing words after 'Impression:'.

2. imbalanced_eval(pre: list, tgt: list, words: list, bucket_size: int) -> None:
    Evaluates the precision and recall of predictions against ground truth data
    by dividing the vocabulary into buckets and calculating metrics for each bucket.

    Parameters:
    - pre (list): A list of predicted words for each sample.
    - tgt (list): A list of ground truth words for each sample.
    - words (list): A list of all unique words in the vocabulary.
    - bucket_size (int): The number of buckets to divide the vocabulary into.

    Returns:
    - None: Prints recall and precision metrics for each bucket.
"""

def preprogress(f):
    f = open(f)
    max_lenn = 76332
    line1 = f.readline()
    gt = []
    i = 0
    while i<max_lenn:
        line1 = line1.split( )
        try:
            if line1[0] == 'Impression:':
                line1 = line1[1:]
                gt.append(line1)
            i += 1
            line1 = f.readline()
        except:
            i += 1
            line1 = f.readline()
            pass
    f.close()
    return gt

# gt = preprogress('/home/ywu10/Documents/multimodel/results/1run_gt_results_2022-05-04-17-20.txt')
# pre = preprogress('/home/ywu10/Documents/multimodel/results/1run_pre_results_2022-05-06-07-28.txt')

def imbalanced_eval(pre,tgt,words):

    recall_ = []
    precision_ = []
    right_ = []
    gap = len(words)//7

    mm = 0
    for index in range(0,len(words),gap):
        mm += 1
        right = 0
        recall = 0
        precision = 0
        for i in range(len(tgt)):
            a = [j for j in tgt[i] if j in words[index:index+gap]]
            b = [j for j in pre[i] if j in words[index:index+gap]]
            right += min(len([j for j in a if j in b]),len([j for j in b if j in a]))
            recall += len(a)
            precision += len(b)
        recall_.append(recall)
        precision_.append(precision)
        right_.append(right)
    print(f'recall:{np.array(right_)/np.array(recall_)}')
    print(f'precision:{np.array(right_)/np.array(precision_)}')
    print(precision_)
    print(recall_)

# imbalanced_eval(pre,gt,words)


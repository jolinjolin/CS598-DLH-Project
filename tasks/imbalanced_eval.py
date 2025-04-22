import json
import numpy as np
import re
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tokenizers import Tokenizer

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

2. imbalanced_eval(pre: list, tgt: list, words: list) -> None:
    Evaluates the precision and recall of predictions against ground truth data
    by dividing the vocabulary into buckets and calculating metrics for each bucket.

    Parameters:
    - pre (list): A list of predicted words for each sample.
    - tgt (list): A list of ground truth words for each sample.
    - words (list): A list of all unique words in the vocabulary.

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
            # if line1[0] == 'Impression:':
            #     line1 = line1[1:]
            #     gt.append(line1)
            gt.append(line1)
            i += 1
            line1 = f.readline()
        except:
            i += 1
            line1 = f.readline()
            pass
    f.close()
    return gt

def imbalanced_eval(pre,tgt,words, num_splits=8):

    recall_ = []
    precision_ = []
    right_ = []
    gap = len(words)//(num_splits-1)

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
    recall_arr = np.array(right_)/np.array(recall_)
    precision_arr = np.array(right_)/np.array(precision_)
    print(f'recall:{recall_arr}')
    print(f'precision:{precision_arr}')
    print(precision_)
    print(recall_)
    if recall_arr[0]is not None and precision_arr[0] is not None and recall_arr[0] != 0 and precision_arr[0] != 0:
        print(f'F1 score (high freq token set): {2*recall_arr[0]*precision_arr[0]/(recall_arr[0]+precision_arr[0])}')
    if recall_arr[-1]is not None and precision_arr[-1] is not None and recall_arr[-1] != 0 and precision_arr[-1] != 0:
        print(f'F1 score (low freq token set): {2*recall_arr[-1]*precision_arr[-1]/(recall_arr[-1]+precision_arr[-1])}')

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/iu_xray/images/",
        help="the path to the directory containing the data.",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        default="data/iu_xray/annotation.json",
        help="the path to the directory containing the data.",
    )
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                    help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_splits', type=int, default=8, help='the number of splits for the words.')
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--pre_path', type=str)

    args = parser.parse_args()

    return args

def main():
    args = parse_agrs()
    gt = preprogress(args.gt_path)
    pre = preprogress(args.pre_path)
    args = parse_agrs()
    tokenizer = Tokenizer(args)
    words = [w for w in tokenizer.token2idx][:-2]
    imbalanced_eval(pre,gt,words, args.num_splits)

if __name__ == "__main__":
    main()


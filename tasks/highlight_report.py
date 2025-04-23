import argparse
from collections import defaultdict, Counter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from utils.tokenizers import Tokenizer

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
    parser.add_argument('--infrequent_ratio', type=float, default=0.8, help='the ratio of infrequent tokens in the vocabulary.')
    parser.add_argument('--pre_path', type=str)
    parser.add_argument('--pre_output_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--gt_output_path', type=str)

    args = parser.parse_args()

    return args

def get_infrequent_token_set(tokenizer, infrequent_ratio):
    tokens = [w for w in tokenizer.token2idx][:-2]
    num_tokens = int(len(tokens) * infrequent_ratio)
    infrequent_token_set = set(tokens[-num_tokens:])
    print(f"Number of infrequent tokens: {len(infrequent_token_set)}")
    return infrequent_token_set

def extract_infrequent_tokens_from_report(df_pre, df_gt, infrequent_token_set, pre_output_path, gt_output_path):
    for i, row in df_gt.iterrows():
        report = row.iloc[0]
        gt_tokens = report.split(" ")
        infreq_tokens, correct_infreq_tokens = [], []
        for gt_token in gt_tokens:
            if gt_token in infrequent_token_set:
                infreq_tokens.append(gt_token)
        for pre_token in df_pre.iloc[i, 0].split(" "):
            if pre_token in infreq_tokens:
                correct_infreq_tokens.append(pre_token)     
        df_gt.at[i, 1] = " ".join(infreq_tokens)
        df_pre.at[i, 1] = " ".join(correct_infreq_tokens)
    df_gt.to_csv(gt_output_path, index=False, header=None)
    df_pre.to_csv(pre_output_path, index=False, header=None)


def main():
    args = parse_agrs()
    tokenizer = Tokenizer(args)
    df_pre = pd.read_csv(args.pre_path, header=None)
    df_gt = pd.read_csv(args.gt_path, header=None)
    infrequent_token_set = get_infrequent_token_set(tokenizer, args.infrequent_ratio)
    extract_infrequent_tokens_from_report(df_pre, df_gt, infrequent_token_set, args.pre_output_path, args.gt_output_path)
    
if __name__ == "__main__":
    main()
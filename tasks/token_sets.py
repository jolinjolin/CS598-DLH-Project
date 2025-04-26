import argparse
from collections import defaultdict, Counter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import spacy
from utils.tokenizers import Tokenizer
import json


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument(
        "--ann_path",
        type=str,
        default="data/iu_xray_raw/annotation.json",
        help="the path to the directory containing the data.",
    )
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                    help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')

    args = parser.parse_args()

    return args
                

def get_sorted_tokens(tokenizer):
    sorted_tokens = [w for w in tokenizer.token2idx][:-2]
    print(f"Vocab size: {len(sorted_tokens)}")
    return sorted_tokens

def get_report_text(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    for split in ["train"]:
        for example in json_data[split]:
            if 'report' in example:
                yield example['report']

def extract_disease_entities(nlp, text):
    medical_terms = set()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            for token in ent.text.split(" "):
                medical_terms.add(token)
    return medical_terms


def split_tokens(sorted_tokens, medical_terms, num_splits=5):
    # # Greedy frequency-balanced binning
    # splits = [[] for _ in range(num_splits)]
    # split_freqs = [0] * num_splits

    # for token, freq in sorted_tokens:
    #     min_index = split_freqs.index(min(split_freqs))
    #     splits[min_index].append(token)
    #     split_freqs[min_index] += freq

    splits = [[] for _ in range(num_splits)]
    num_tokens_per_split = len(sorted_tokens) // num_splits
    for i in range(num_splits):
        start_index = i * num_tokens_per_split
        end_index = (i + 1) * num_tokens_per_split if i != num_splits - 1 else len(sorted_tokens)
        splits[i] = [token for token in sorted_tokens[start_index:end_index]]

    ratios = []
    for i, split in enumerate(splits):
        num_medical = 0
        for token in split:
            if token in medical_terms:
                num_medical += 1
        print(f"Split {i+1}:")
        print(f"Num of medical terms: {num_medical}")
        print(f"Num of all terms: {len(split)}")
        ratios.append(num_medical / len(split))
    print(f"Ratios of meical terms: {ratios}")

def main():
    args = parse_agrs()
    tot_num_tokens = 0
    reports_txt= get_report_text(args.ann_path)
    medical_terms_all = set()
    nlp = spacy.load("en_ner_bc5cdr_md")
    for report in reports_txt:
        tot_num_tokens += len(report.split())
        medical_terms = extract_disease_entities(nlp, report)
        medical_terms_all.update(medical_terms)
    print(f"Tot num of medical terms: {len(medical_terms_all)}")
    print(f"Tot num of tokens: {tot_num_tokens}")

    tokenizer = Tokenizer(args) 
    sorted_tokens = get_sorted_tokens(tokenizer)
    split_tokens(sorted_tokens, medical_terms_all, num_splits=5)

    
if __name__ == "__main__":
    main()
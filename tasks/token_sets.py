import argparse
from collections import defaultdict, Counter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import spacy
from spacy.tokens import Doc
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

    args = parser.parse_args()

    return args

def get_sorted_tokens(tokenizer):
    total_tokens = []
    report_length, tot_report = 0, 0
    for example in tokenizer.ann['train']:
        tokens = tokenizer.clean_report(example['report']).split()
        total_tokens.extend(tokens)
        report_length += len(tokens)
        tot_report += 1
    
    # for example in tokenizer.ann['val']:
    #     tokens = tokenizer.clean_report(example['report']).split()
    #     total_tokens.extend(tokens)
    #     report_length += len(tokens)
    #     tot_report += 1
    
    # for example in tokenizer.ann['test']:
    #     tokens = tokenizer.clean_report(example['report']).split()
    #     total_tokens.extend(tokens)
    #     report_length += len(tokens)
    #     tot_report += 1

    counter = Counter(total_tokens)
    
    vocab_freqs = {tok: freq for tok, freq in counter.items() if freq >= tokenizer.threshold}
    vocab_freqs['<unk>'] = 1e-5  # Give <unk> a tiny freq to keep it

    sorted_tokens = sorted(vocab_freqs.items(), key=lambda x: x[1])

    print(f"Avg report length: {report_length / tot_report}")
    print(f"Vocab size: {len(sorted_tokens)}")
    return sorted_tokens


def extract_disease_entities(text):
    nlp = spacy.load("en_ner_bc5cdr_md")
    doc = nlp(text)
    return len(doc.ents)


def split_tokens(sorted_tokens, num_splits=5):
    # Greedy frequency-balanced binning
    splits = [[] for _ in range(num_splits)]
    split_freqs = [0] * num_splits

    for token, freq in sorted_tokens:
        min_index = split_freqs.index(min(split_freqs))
        splits[min_index].append(token)
        split_freqs[min_index] += freq

    # splits = [[] for _ in range(num_splits)]
    # num_tokens_per_split = len(sorted_tokens) // num_splits
    # for i in range(num_splits):
    #     start_index = i * num_tokens_per_split
    #     end_index = (i + 1) * num_tokens_per_split if i != num_splits - 1 else len(sorted_tokens)
    #     splits[i] = [token for token,_ in sorted_tokens[start_index:end_index]]

    ratios = []
    for i, split in enumerate(splits):
        num_medical = extract_disease_entities(" ".join(split))
        print(f"Split {i+1}:")
        print(f"Num of medical terms: {num_medical}")
        print(f"Num of all terms: {len(split)}")
        ratios.append(num_medical / len(split))
    print(f"Ratios of meical terms: {ratios}")

def main():
    args = parse_agrs()
    tokenizer = Tokenizer(args)
    sorted_tokens = get_sorted_tokens(tokenizer)
    split_tokens(sorted_tokens, num_splits=5)

    
if __name__ == "__main__":
    main()
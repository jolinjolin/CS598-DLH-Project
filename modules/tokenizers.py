import json
import re
from collections import Counter

"""
Tokenizer class for processing and tokenizing medical reports.

Attributes:
    ann_path (str): Path to the annotation file containing training data.
    threshold (int): Minimum frequency for a token to be included in the vocabulary.
    dataset_name (str): Name of the dataset ('iu_xray' or other).
    clean_report (function): Function to clean reports based on the dataset type.
    ann (dict): Parsed JSON data from the annotation file.
    token2idx (dict): Mapping from tokens to their corresponding indices.
    idx2token (dict): Mapping from indices to their corresponding tokens.

Methods:
    create_vocabulary():
        Creates a vocabulary from the training data based on token frequency.
        Returns:
            tuple: A tuple containing token2idx and idx2token dictionaries.

    create_vocabulary2():
        Alternative method to create a vocabulary, sorting tokens by frequency.
        Returns:
            tuple: A tuple containing token2idx and idx2token dictionaries.

    clean_report_iu_xray(report):
        Cleans and preprocesses a report specific to the 'iu_xray' dataset.
        Args:
            report (str): The report to be cleaned.
        Returns:
            str: The cleaned report.

    clean_report_mimic_cxr(report):
        Cleans and preprocesses a report specific to the 'mimic_cxr' dataset.
        Args:
            report (str): The report to be cleaned.
        Returns:
            str: The cleaned report.

    get_token_by_id(id):
        Retrieves the token corresponding to a given index.
        Args:
            id (int): The index of the token.
        Returns:
            str: The corresponding token.

    get_id_by_token(token):
        Retrieves the index corresponding to a given token.
        Args:
            token (str): The token to look up.
        Returns:
            int: The corresponding index, or the index of '<unk>' if not found.

    get_vocab_size():
        Retrieves the size of the vocabulary.
        Returns:
            int: The size of the vocabulary.

    __call__(report):
        Tokenizes a given report into a sequence of token indices.
        Args:
            report (str): The report to tokenize.
        Returns:
            list: A list of token indices, including start and end tokens.

    decode(ids):
        Decodes a sequence of token indices back into a string.
        Args:
            ids (list): A list of token indices.
        Returns:
            str: The decoded string.

    decode_batch(ids_batch):
        Decodes a batch of sequences of token indices back into strings.
        Args:
            ids_batch (list of lists): A batch of lists of token indices.
        Returns:
            list: A list of decoded strings.
"""

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        #self.token2idx, self.idx2token = self.create_vocabulary()
        self.token2idx, self.idx2token = self.create_vocabulary2()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token


    def create_vocabulary2(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = dict(sorted(counter.items(),key = lambda x:x[1],reverse=True))
        vocab =  [i  for  i in vocab if  vocab[i]>=self.threshold]
        vocab.append('<unk>')
        #vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report


    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

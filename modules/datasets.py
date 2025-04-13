import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

"""
This module defines dataset classes for handling medical image datasets with associated reports.

Classes:
    BaseDataset:
        A base class for datasets that handle medical images and their associated textual reports.
        Attributes:
            image_dir (str): Directory containing the images.
            ann_path (str): Path to the annotation file (JSON format).
            max_seq_length (int): Maximum sequence length for tokenized reports.
            split (str): Dataset split (e.g., 'train', 'val', 'test').
            tokenizer (callable): Tokenizer function for processing textual reports.
            transform (callable, optional): Transformations to apply to images.
            ann (dict): Parsed annotations from the JSON file.
            examples (list): List of examples for the specified split.
        Methods:
            __len__(): Returns the number of examples in the dataset.

    IuxrayMultiImageDataset(BaseDataset):
        A dataset class for IU X-Ray data with multiple images per example.
        Methods:
            __getitem__(idx): Retrieves a sample consisting of two images, tokenized report, and metadata.

    MimiccxrSingleImageDataset(BaseDataset):
        A dataset class for MIMIC-CXR data with a single image per example.
        Methods:
            __getitem__(idx): Retrieves a sample consisting of one image, tokenized report, and metadata.
"""

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

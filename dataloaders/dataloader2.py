import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets2 import IuxrayMultiImageDataset, MimiccxrSingleImageDataset

"""
R2DataLoader Class

A custom DataLoader class for handling datasets and data loading for medical image datasets.

Attributes:
    args (Namespace): Arguments containing dataset configurations.
    tokenizer (object): Tokenizer for processing text data.
    split (str): Dataset split ('train', 'val', or 'test').
    shuffle (bool): Whether to shuffle the data.
    dataset_name (str): Name of the dataset ('iu_xray' or other).
    batch_size (int): Number of samples per batch.
    num_workers (int): Number of subprocesses to use for data loading.
    transform (torchvision.transforms.Compose): Transformations applied to the images.
    dataset (Dataset): Dataset object for the specified dataset.
    init_kwargs (dict): Keyword arguments for initializing the DataLoader.

Methods:
    __init__(args, tokenizer, split, shuffle):
        Initializes the R2DataLoader with the specified arguments and configurations.

    collate_fn(data):
        Custom collate function for batching data.
        
        Args:
            data (list): List of samples from the dataset.
        
        Returns:
            tuple: Batched data including:
                - images_id (list): List of image IDs for the first dataset.
                - images (torch.Tensor): Batched images for the first dataset.
                - targets (torch.LongTensor): Batched target sequences for the first dataset.
                - targets_masks (torch.FloatTensor): Batched target masks for the first dataset.
                - images_id (list): List of image IDs for the first dataset (repeated).
                - images_id2 (list): List of image IDs for the second dataset.
                - images2 (torch.Tensor): Batched images for the second dataset.
                - targets2 (torch.LongTensor): Batched target sequences for the second dataset.
                - targets_masks2 (torch.FloatTensor): Batched target masks for the second dataset.
                - images_id2 (list): List of image IDs for the second dataset (repeated).
"""

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):

        #images_id, images, reports_ids, reports_masks, seq_lengths, neg = zip(*data)
        sample,sample2  = zip(*data)
        images_id, images, reports_ids, reports_masks, seq_lengths, neg = zip(*sample)
        images_id2, images2, reports_ids2, reports_masks2, seq_lengths2, neg2 = zip(*sample2)

        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        #------

        images2 = torch.stack(images2, 0)
        max_seq_length2 = max(seq_lengths2)

        targets2 = np.zeros((len(reports_ids2), max_seq_length2), dtype=int)
        targets_masks2 = np.zeros((len(reports_ids2), max_seq_length2), dtype=int)

        for i, report_ids2 in enumerate(reports_ids2):
            targets2[i, :len(report_ids2)] = report_ids2

        for i, report_masks2 in enumerate(reports_masks2):
            targets_masks2[i, :len(report_masks2)] = report_masks2


        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks),images_id , \
               images_id2, images2, torch.LongTensor(targets2), torch.FloatTensor(targets_masks2),images_id2 #  torch.cat(neg,0)



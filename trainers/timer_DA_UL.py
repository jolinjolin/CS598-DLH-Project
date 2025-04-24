import logging
import os
from abc import abstractmethod
import numpy as np
import torch
from numpy import inf
import torch.nn.functional as F
from .trainer_base import BaseTrainer
from tqdm import tqdm
from collections import Counter


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()

        # Define static frequent token set (Uh)
        # Example: top 100 most frequent tokens (excluding PAD, BOS, EOS)
        freq_tokens = list(self.model.tokenizer.token2idx.values())[:100]

        for batch_idx, (images_id, images, reports_ids, reports_masks, _,
                        images_id2, images2, reports_ids2, reports_masks2, _) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)

            # Forward pass
            output = self.model(images, reports_ids, mode='train')

            # Compute NLL (cross-entropy loss)
            nll_loss = self.criterion(output, reports_ids, reports_masks)

            # Compute Unlikelihood Loss (UL) with static token set
            negative_word = self.build_negative_word_tensor(output, freq_tokens)  # Create tensor for frequent tokens
            ul_loss = self.unlikelihood_loss(output, negative_word, reports_masks)

            # Total loss = NLL + UL
            loss = nll_loss + ul_loss
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # Validation & test steps (same as before)
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _,
                            images_id2, images2, reports_ids2, reports_masks2, _) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        # Test
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _,
                            images_id2, images2, reports_ids2, reports_masks2, _) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()
        return log

    def reward(self, tgt, pre):
        words = [w for w in self.model.tokenizer.token2idx][:-2]
        recall_ = []
        precision_ = []
        right_ = []
        gap = len(words) // 8
        for index in range(0, len(words) - gap, gap):
            right = 0
            recall = 0
            precision = 0
            for i in range(len(tgt)):
                a = [j for j in tgt[i].split() if j in words[index:index + gap]]
                b = [j for j in pre[i].split() if j in words[index:index + gap]]
                right += len([j for j in a if j in b])
                recall += len(a)
                precision += len(b)
            recall_.append(recall)
            precision_.append(precision)
            right_.append(right)
        recall = np.array(right_) / np.array(recall_)
        precision = np.array(right_) / np.array(precision_)
        score = 2 * precision * recall / (precision + recall)
        return np.sum(np.nan_to_num(score))

    def imbalanced_eval(self, pre, tgt, n=8):
        # Directly get the n least frequent tokens (excluding the last two special tokens)
        words = [w for w in self.model.tokenizer.token2idx][-n:-2]

        recall_ = []
        precision_ = []
        right_ = []
        gap = len(words) // n  # This will typically be 1 if you're just using the last n tokens

        for index in range(0, len(words) - gap + 1, gap):  # Adjust to ensure last token group is covered
            right = 0
            recall = 0
            precision = 0
            for i in range(len(tgt)):
                a = [j for j in tgt[i].split() if j in words[index:index + gap]]
                b = [j for j in pre[i].split() if j in words[index:index + gap]]
                right += len([j for j in a if j in b])
                recall += len(a)
                precision += len(b)
            recall_.append(recall)
            precision_.append(precision)
            right_.append(right)

        recall_arr = np.array(right_) / np.array(recall_)
        precision_arr = np.array(right_) / np.array(precision_)

        print(f"recall: {recall_arr}")
        print(f"precision: {precision_arr}")
        print(f"precision counts: {precision_}")
        print(f"recall counts: {recall_}")

    def unlikelihood_loss(self, output, negative_word, mask):

        output = torch.sum(mask[:, 1:].unsqueeze(-1) * output, dim=1) / torch.sum(mask)
        loss = -torch.log(torch.clamp(1.0 - (output * negative_word).exp(), min=1e-20))
        return torch.mean(torch.mean(loss, dim=-1), dim=-1)

    def build_negative_word_tensor(self, output, freq_tokens):
        # output shape: [batch_size, seq_len, vocab_size]
        vocab_size = output.size(-1)

        # Create a binary mask: [vocab_size]
        negative_word = output.new_zeros(vocab_size)  # [vocab_size]
        negative_word[freq_tokens] = 1  # Set frequent tokens to 1

        # Expand dimensions: [1, 1, vocab_size] for broadcasting
        negative_word = negative_word.unsqueeze(0).unsqueeze(0)

        return negative_word  # Will broadcast across batch and sequence length

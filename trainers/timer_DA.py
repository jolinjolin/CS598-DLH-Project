import logging
import os
from abc import abstractmethod
import numpy as np
import torch
from numpy import inf
import torch.nn.functional as F
from .trainer_base import BaseTrainer

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

        for batch_idx, (images_id, images, reports_ids, reports_masks, _,
                        images_id2, images2, reports_ids2, reports_masks2, _) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)

            # Standard forward pass
            output = self.model(images, reports_ids, mode='train')

            # Compute only negative log-likelihood (cross-entropy) loss
            loss = self.criterion(output, reports_ids, reports_masks)

            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # Validation
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

    def imbalanced_eval(self, pre, tgt, n):

        # words = dict(sorted(dict(self.model.tokenizer.counter).items(), key=lambda x: x[1]))
        words = [w for w in self.model.tokenizer.token2idx][:-2]
        recall_ = []
        precision_ = []
        right_ = []
        gap = len(words) // n
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
        print(f'recall:{np.array(right_) / np.array(recall_)}')
        print(f'precision:{np.array(right_) / np.array(precision_)}')
        print(precision_)
        print(recall_)

    def unlikelihood_loss(self, output, negative_word, mask):

        output = torch.sum(mask[:, 1:].unsqueeze(-1) * output, dim=1) / torch.sum(mask)
        loss = -torch.log(torch.clamp(1.0 - (output * negative_word).exp(), min=1e-20))
        return torch.mean(torch.mean(loss, dim=-1), dim=-1)

    def _train_epoch2(self, epoch):

        train_loss = 0
        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            # self.imbalanced_eval(val_res,val_gts)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))

        self.model.eval()
        with torch.no_grad():

            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            # test_gts, test_res = self.heihei()
            self.imbalanced_eval(test_res, test_gts)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            print(test_met)
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log
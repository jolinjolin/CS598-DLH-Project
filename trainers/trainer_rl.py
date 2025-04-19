import numpy as np
import torch
from numpy import inf
from tqdm import tqdm
import torch.nn.functional as F
from .trainer_base import BaseTrainer

np.seterr(divide="ignore", invalid="ignore")

"""
This module contains the implementation of `TrainerRL` classes for training deep learning models
with support for meta-learning and reinforcement learning optimization.
Classes:
    - TrainerRL: Concrete implementation of `BaseTrainer` that includes specific training logic for a model with
      meta-learning and reinforcement learning components.
TrainerRL (inherits from BaseTrainer):
    Methods:
        - __init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
            Initializes the trainer with additional dataloaders for training, validation, and testing.
        - _train_epoch(epoch):
            Implements the training logic for a single epoch, including meta-learning and reinforcement learning
            optimization.
        - reward(tgt, pre):
            Computes a reward score based on recall and precision for specific word groups.
        - imbalanced_eval(pre, tgt, n):
            Evaluates the model's performance on imbalanced data by calculating recall and precision for word groups.
        - unlikelihood_loss(output, negative_word, mask):
            Computes the unlikelihood loss to penalize undesired predictions.
Notes:
    - The `TrainerRL` class includes advanced training techniques such as meta-learning and reinforcement learning
      for optimizing the model.
"""


class TrainerRL(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        args,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ):
        super(TrainerRL, self).__init__(
            model, criterion, metric_ftns, optimizer, args, lr_scheduler
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        train_loss = 0

        self.model.train()
        count = 0
        for (
            images_id,
            images,
            reports_ids,
            reports_masks,
            _,
            images_id2,
            images2,
            reports_ids2,
            reports_masks2,
            _,
        ) in tqdm(self.train_dataloader):
            images, reports_ids, reports_masks = (
                images.to(self.device),
                reports_ids.to(self.device),
                reports_masks.to(self.device),
            )

            images2, reports_ids2, reports_masks2 = (
                images2.to(self.device),
                reports_ids2.to(self.device),
                reports_masks2.to(self.device),
            )

            origin_param = self.model.state_dict()
            output = self.model(images, reports_ids, mode="train")
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            lm_param = self.model.state_dict()

            # optimize meta-RL
            self.optimizer.zero_grad()
            # self.metarl_opt.zero_grad()
            self.model.load_state_dict(origin_param)
            output = torch.mean(output, dim=1)
            reports_ids_ = (
                torch.sum(
                    F.one_hot(reports_ids, len(self.model.tokenizer.idx2token) + 1),
                    dim=1,
                )
                > 0
            ).long()
            action, entropy = self.metaRL.predict_action(output * reports_ids_)
            output = self.model(images, reports_ids, mode="train")
            loss = self.criterion(
                output, reports_ids, reports_masks
            ) + self.unlikelihood_loss(output, action, reports_masks)

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            rl_param = self.model.state_dict()

            # optimizing rl by calculating reward on test set
            with torch.no_grad():
                self.model.load_state_dict(lm_param)
                output = self.model(images2, mode="sample")
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(
                    reports_ids2[:, 1:].cpu().numpy()
                )
                lm_reward = self.reward(ground_truths, reports)

                self.model.load_state_dict(rl_param)
                output = self.model(images2, mode="sample")
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(
                    reports_ids2[:, 1:].cpu().numpy()
                )
                rl_reward = torch.tensor(self.reward(ground_truths, reports)).to(
                    self.device
                )

            self.model.load_state_dict(origin_param)
            self.metarl_opt.zero_grad()
            output = self.model(images, reports_ids, mode="train")
            output = torch.mean(output, dim=1)
            reports_ids_ = (
                torch.sum(
                    F.one_hot(reports_ids, len(self.model.tokenizer.idx2token) + 1),
                    dim=1,
                )
                > 0
            ).long()
            action, entropy = self.metaRL.predict_action(output * reports_ids_)
            predict_reward = self.metaRL.predict_reward(output * reports_ids_, action)
            output = self.model(images, reports_ids, mode="train")
            loss = self.criterion(
                output, reports_ids, reports_masks
            ) + self.unlikelihood_loss(output, action, reports_masks)

            reward = rl_reward - lm_reward
            loss = (
                (reward - predict_reward) ** 2
                - torch.mean((reward - predict_reward) * loss, dim=0)
                - 0.1 * torch.mean(entropy)
            )
            loss.backward()
            self.metarl_opt.step()

            if reward >= 0:
                self.model.load_state_dict(rl_param)
                count += 1
            else:
                self.model.load_state_dict(lm_param)

        print(f"rl train:{count / len(self.train_dataloader)}")

        log = {"train_loss": train_loss / len(self.train_dataloader)}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (
                images_id,
                images,
                reports_ids,
                reports_masks,
                _,
                images_id2,
                images2,
                reports_ids2,
                reports_masks2,
                _,
            ) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = (
                    images.to(self.device),
                    reports_ids.to(self.device),
                    reports_masks.to(self.device),
                )
                output = self.model(images, mode="sample")
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(
                    reports_ids[:, 1:].cpu().numpy()
                )
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )
            log.update(**{"val_" + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (
                images_id,
                images,
                reports_ids,
                reports_masks,
                _,
                images_id2,
                images2,
                reports_ids2,
                reports_masks2,
                _,
            ) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = (
                    images.to(self.device),
                    reports_ids.to(self.device),
                    reports_masks.to(self.device),
                )
                output = self.model(images, mode="sample")
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(
                    reports_ids[:, 1:].cpu().numpy()
                )
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            self.imbalanced_eval(test_res, test_gts)
            test_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(test_gts)},
                {i: [re] for i, re in enumerate(test_res)},
            )
            log.update(**{"test_" + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()
        print(f"predict:{test_res[-1]}")
        print(f"ground_truth:{test_gts[-1]}")

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
                a = [j for j in tgt[i].split() if j in words[index : index + gap]]
                b = [j for j in pre[i].split() if j in words[index : index + gap]]
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
                a = [j for j in tgt[i].split() if j in words[index : index + gap]]
                b = [j for j in pre[i].split() if j in words[index : index + gap]]
                right += len([j for j in a if j in b])
                recall += len(a)
                precision += len(b)
            recall_.append(recall)
            precision_.append(precision)
            right_.append(right)
        print(f"recall:{np.array(right_) / np.array(recall_)}")
        print(f"precision:{np.array(right_) / np.array(precision_)}")
        print(precision_)
        print(recall_)

    def unlikelihood_loss(self, output, negative_word, mask):
        output = torch.sum(mask[:, 1:].unsqueeze(-1) * output, dim=1) / torch.sum(mask)
        loss = -torch.log(torch.clamp(1.0 - (output * negative_word).exp(), min=1e-20))
        return torch.mean(torch.mean(loss, dim=-1), dim=-1)
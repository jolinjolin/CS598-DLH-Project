import numpy as np
import torch
from numpy import inf
from tqdm import tqdm
import torch.nn.functional as F
from .trainer_base import BaseTrainer

np.seterr(divide="ignore", invalid="ignore")

"""
This module contains the implementation of `TrainerNonRL` classes for training deep learning models
without reinforcement learning optimization.
Classes:
    - TrainerNonRL: Concrete implementation of `BaseTrainer` that includes specific training logic for a model without
      reinforcement learning components.
TrainerNonRL (inherits from BaseTrainer):
    Methods:
        - __init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
            Initializes the trainer with additional dataloaders for training, validation, and testing.
        - _train_epoch(epoch):
            Training logic for a single epoch without meta-learning and reinforcement learning.
"""


class TrainerNonRL(BaseTrainer):
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
        super(TrainerNonRL, self).__init__(
            model, criterion, metric_ftns, optimizer, args, lr_scheduler
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        train_loss = 0
        self.logger.info(
            "[{}/{}] Start to train in the training set.".format(epoch, self.epochs)
        )
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, _,_,_,_,_,_) in enumerate(
            self.train_dataloader
        ):
            images, reports_ids, reports_masks = (
                images.to(self.device),
                reports_ids.to(self.device),
                reports_masks.to(self.device),
            )
            output = self.model(images, reports_ids, mode="train")
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info(
                    "[{}/{}] Step: {}/{}, Training Loss: {:.5f}.".format(
                        epoch,
                        self.epochs,
                        batch_idx,
                        len(self.train_dataloader),
                        train_loss / (batch_idx + 1),
                    )
                )
        log = {"train_loss": train_loss / len(self.train_dataloader)}

        self.logger.info(
            "[{}/{}] Start to evaluate in the validation set.".format(
                epoch, self.epochs
            )
        )
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _,_,_,_,_,_) in enumerate(
                self.val_dataloader
            ):
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
            # self.imbalanced_eval(val_res,val_gts)
            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )
            log.update(**{"val_" + k: v for k, v in val_met.items()})

        self.logger.info(
            "[{}/{}] Start to evaluate in the test set.".format(epoch, self.epochs)
        )

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _,_,_,_,_,_) in enumerate(
                self.test_dataloader
            ):
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

            # test_gts, test_res = self.heihei()
            # self.imbalanced_eval(test_res, test_gts)
            test_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(test_gts)},
                {i: [re] for i, re in enumerate(test_res)},
            )
            print(test_met)
            log.update(**{"test_" + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log
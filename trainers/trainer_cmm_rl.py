import torch
import numpy as np
from .rewards import get_self_critical_reward, init_scorer
from .loss import compute_loss
from .trainer_base import BaseTrainer

np.seterr(divide="ignore", invalid="ignore")

"""
Trainer class for training and evaluating a model with self-critical reinforcement learning.

Attributes:
    train_dataloader (DataLoader): DataLoader for the training dataset.
    val_dataloader (DataLoader): DataLoader for the validation dataset.
    test_dataloader (DataLoader): DataLoader for the test dataset.

Methods:
    - __init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
            Initializes the trainer with additional dataloaders for training, validation, and testing.
    _train_epoch(epoch):
        Trains the model for one epoch, evaluates it on the validation and test datasets, and logs the results.
"""
class TrainerCMMRL(BaseTrainer):
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
        super(TrainerCMMRL, self).__init__(
            model, criterion, metric_ftns, optimizer, args, lr_scheduler
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)

            # ********* Self-Critical *********
            init_scorer()
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(images, mode='sample',
                                           update_opts={'sample_method': self.args.sc_sample_method,
                                                        'beam_size': self.args.sc_beam_size})

            self.model.train()
            gen_result, sample_logprobs = self.model(images, mode='sample',
                                                     update_opts={'sample_method': self.args.train_sample_method,
                                                                  'beam_size': self.args.train_beam_size,
                                                                  'sample_n': self.args.train_sample_n})

            gts = reports_ids[:, 1:]
            reward = get_self_critical_reward(greedy_res, gts, gen_result)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss_rl = self.criterion(sample_logprobs, gen_result.data, reward)

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train')
            loss_nll = compute_loss(output, reports_ids, reports_masks)

            loss = 0.01 * loss_nll + 0.99 * loss_rl

            # ********* Self-Critical *********

            train_loss += loss.item()
            # self.ve_optimizer.zero_grad()
            # self.ed_optimizer.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            # self.ve_optimizer.step()
            # self.ed_optimizer.step()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                lrs = self._get_learning_rate()
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.6f}, LR (ve): {:.6f}, LR (ed): {:6f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1), lrs['lr_visual_extractor'],
                                         lrs['lr_encoder_decoder']))

            if (batch_idx+1) % self.args.sc_eval_period == 0:
                log = {'train_loss': train_loss / (batch_idx + 1)}

                self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
                self.model.eval()
                with torch.no_grad():
                    # val_loss = 0
                    val_gts, val_res = [], []
                    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                            self.device), reports_masks.to(self.device)

                        # # ****** Compute Loss ******
                        # images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                        #                                      reports_masks.to(self.device)
                        # output = self.model(images, reports_ids, mode='train')
                        # loss = self.criterion(output, reports_ids, reports_masks)
                        # val_loss += loss.item()
                        # # ****** Compute Loss ******

                        output, _ = self.model(images, mode='sample')
                        reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        val_res.extend(reports)
                        val_gts.extend(ground_truths)

                        # for id, re, gt in zip(images_id, reports, ground_truths):
                        #     print(id)
                        #     print('[Generated]: {}'.format(re))
                        #     print('[Ground Truth]: {}'.format(gt))

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

                        # for id, re, gt in zip(images_id, reports, ground_truths):
                        #     print(id)
                        #     print('[Generated]: {}'.format(re))
                        #     print('[Ground Truth]: {}'.format(gt))

                    test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                                {i: [re] for i, re in enumerate(test_res)})
                    log.update(**{'test_' + k: v for k, v in test_met.items()})
                self._save_best(epoch, log)
                self._print_to_file(log)
                self._write_to_file(test_gts, test_res, epoch, batch_idx)

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            # val_loss = 0
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                # # ****** Compute Loss ******
                # images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                #                                      reports_masks.to(self.device)
                # output = self.model(images, reports_ids, mode='train')
                # loss = self.criterion(output, reports_ids, reports_masks)
                # val_loss += loss.item()
                # # ****** Compute Loss ******

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

                for id, re, gt in zip(images_id, reports, ground_truths):
                    print(id)
                    print('[Generated]: {}'.format(re))
                    print('[Ground Truth]: {}'.format(gt))

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            # log.update(**{'val_loss': val_loss / len(self.val_dataloader)})

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

                for id, re, gt in zip(images_id, reports, ground_truths):
                    print(id)
                    print('[Generated]: {}'.format(re))
                    print('[Ground Truth]: {}'.format(gt))

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        log.update(**self._get_learning_rate())
        self._write_to_file(test_gts, test_res, epoch, 0)
        # self.lr_scheduler.step()

        return log
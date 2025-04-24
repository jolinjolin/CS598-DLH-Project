import logging
import os
from abc import abstractmethod
import numpy as np
import torch
from numpy import inf
from tqdm import tqdm
from models.metalearning_model import MetaLearningModel
import torch.nn.functional as F

np.seterr(divide="ignore", invalid="ignore")

"""
This module contains the implementation of the `BaseTrainer`
Classes:
    - BaseTrainer: Abstract base class for training models, providing common functionality such as checkpointing,
      device preparation, and training loop management.
BaseTrainer:
    Methods:
        - __init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler):
            Initializes the trainer with the given model, loss function, metrics, optimizer, arguments, and learning
            rate scheduler.
        - _train_epoch(epoch):
            Abstract method to be implemented by subclasses for training logic in a single epoch.
        - train():
            Executes the training loop over the specified number of epochs, including validation and checkpointing.
        - _record_best(log):
            Records the best validation and test metrics during training.
        - _print_best():
            Logs the best validation and test results.
        - _prepare_device(n_gpu_use):
            Prepares the device (CPU/GPU) for training and returns the device and list of GPU IDs.
        - _save_checkpoint(epoch, save_best=False):
            Saves the current model checkpoint and optionally saves the best model checkpoint.
        - _resume_checkpoint(resume_path):
            Resumes training from a saved checkpoint.
    Notes:
    - The `BaseTrainer` class is designed to be extended for specific training implementations.
"""
class BaseTrainer(object):

    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.metaRL = MetaLearningModel(model.tokenizer).to(self.device)
        self.metarl_opt = torch.optim.Adam(
            self.metaRL.parameters(), lr=0.001, betas=(0.9, 0.99), eps=0.0000001
        )

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = "val_" + args.monitor_metric
        self.mnt_metric_test = "test_" + args.monitor_metric
        assert self.mnt_mode in ["min", "max"]

        self.mnt_best = inf if self.mnt_mode == "min" else -inf
        self.early_stop = getattr(self.args, "early_stop", inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {
            "val": {self.mnt_metric: self.mnt_best},
            "test": {self.mnt_metric_test: self.mnt_best},
        }

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            args.resume = "./results/iu_xray/model_best.pth"
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("\t{:15s}: {}".format(str(key), value))
            # break
            """
            if self.best_recorder['test']['test_BLEU_1']>log['test_BLEU_1']:
                self._save_checkpoint(epoch, save_best=True)
            else:
                self._save_checkpoint(epoch, save_best=False)
            """
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        improved_val = (
            self.mnt_mode == "min"
            and log[self.mnt_metric] <= self.best_recorder["val"][self.mnt_metric]
        ) or (
            self.mnt_mode == "max"
            and log[self.mnt_metric] >= self.best_recorder["val"][self.mnt_metric]
        )
        if improved_val:
            self.best_recorder["val"].update(log)

        improved_test = (
            self.mnt_mode == "min"
            and log[self.mnt_metric_test]
            <= self.best_recorder["test"][self.mnt_metric_test]
        ) or (
            self.mnt_mode == "max"
            and log[self.mnt_metric_test]
            >= self.best_recorder["test"][self.mnt_metric_test]
        )
        if improved_test:
            self.best_recorder["test"].update(log)

    def _print_best(self):
        self.logger.info(
            "Best results (w.r.t {}) in validation set:".format(
                self.args.monitor_metric
            )
        )
        for key, value in self.best_recorder["val"].items():
            self.logger.info("\t{:15s}: {}".format(str(key), value))

        self.logger.info(
            "Best results (w.r.t {}) in test set:".format(self.args.monitor_metric)
        )
        for key, value in self.best_recorder["test"].items():
            self.logger.info("\t{:15s}: {}".format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device("cuda:0" if n_gpu_use > 0 else "mps")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
        }
        filename = os.path.join(self.checkpoint_dir, "current-checkpoint.pth")
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, "best-model.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        # resume_path = './models/model_iu_xray.pth'
        # resume_path = './models/model_mimic_cxr.pth'
        resume_path = "./results/iu_xray/best-model.pth"
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
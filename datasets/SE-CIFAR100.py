# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms

from backbone.MNISTMLP import MNISTMLP
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from seq_cifar100 import TCIFAR100, MyCIFAR100
from transforms.embed import R18_EXTRACTOR, BatchTransformDataLoader
from utils.conf import base_path_dataset as base_path


class SplitEmbeddedCIFAR100(ContinualDataset):

    NAME = 'SE-CIFAR100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408),
                              (0.2675, 0.2565, 0.2761))])

    def __init__(self, args):
        super().__init__(args)

        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path(), train=True,
                                   download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path(), train=False,
                                     download=True, transform=test_transform)

        # Support shuffling the task composition
        train_dataset.targets = [self.substitution_table[y]
                                 for y in train_dataset.targets]
        test_dataset.targets = [self.substitution_table[y]
                                for y in test_dataset.targets]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def get_data_loaders(self):
        train, test = store_masked_loaders(
            self.train_dataset, self.test_dataset, self)
        train = BatchTransformDataLoader(
            train, lambda x, y, original_x: (R18_EXTRACTOR(x), y, original_x))
        test = BatchTransformDataLoader(
            test, lambda x, y: (R18_EXTRACTOR(x), y))

        # Hack to override the `store_masked_loaders` function
        self.test_loaders[-1] = test
        self.train_loader = train
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SplitEmbeddedCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return MNISTMLP(512,
                        SplitEmbeddedCIFAR100.N_CLASSES_PER_TASK * SplitEmbeddedCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SplitEmbeddedCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(
        ), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

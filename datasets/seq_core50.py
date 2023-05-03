import avalanche.benchmarks.datasets as avl_datasets

from typing import List, Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class SplitCORe50(avl_datasets.CORe50Dataset):
    """
    Overrides the CORe50Dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, mini=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(SplitCORe50, self).__init__(
            root=root, 
            train=train,
            transform=transform,
            target_transform=target_transform,
            mini=mini)

    @property
    def data(self) -> List[str]:
        return self.paths
    
    @data.setter
    def data(self, value) -> List[str]:
        self.paths = value

    def __getitem__(self, index):
        target = self.targets[index]
        if self.mini:
            bp = "core50_32x32"
        else:
            bp = "core50_128x128"

        img = self.loader(str(self.root / bp / self.paths[index]))
        original_img = self.not_aug_transform(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, original_img


class SequentialCORe50(ContinualDataset):

    NAME = 'seq-core50'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 10
    NORMALIZATION_CONSTANTS = ([0.4802, 0.4481, 0.3975], [0.2770, 0.2690, 0.2821])
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(*NORMALIZATION_CONSTANTS)])

    def __init__(self, args) -> None:
        super().__init__(args)

        transform = self.TRANSFORM
        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])
        train_dataset = SplitCORe50(base_path(), train=True,
                                  download=True, transform=transform, mini=True)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = avl_datasets.CORe50Dataset(base_path(), train=False,
                                   download=True, transform=test_transform, mini=True)

        # Support shuffling the task composition
        train_dataset.targets = [self.substitution_table[y] for y in train_dataset.targets]
        test_dataset.targets  = [self.substitution_table[y] for y in test_dataset.targets]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    def get_data_loaders(self):
        train, test = store_masked_loaders(self.train_dataset, self.test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCORe50.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCORe50.N_CLASSES_PER_TASK
                        * SequentialCORe50.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(*SequentialCORe50.NORMALIZATION_CONSTANTS)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(*SequentialCORe50.NORMALIZATION_CONSTANTS)

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCORe50.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler


import avalanche.benchmarks.datasets as avl_datasets
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.MNISTMLP import MNISTMLP

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from seq_core50 import SequentialCORe50, SplitCORe50
from transforms.embed import R18_EXTRACTOR, BatchTransformDataLoader
from utils.conf import base_path_dataset as base_path


class SplitEmbeddedCORe50(ContinualDataset):

    NAME = 'SE-CORe50'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 10
    NORMALIZATION_CONSTANTS = SequentialCORe50.NORMALIZATION_CONSTANTS
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(*NORMALIZATION_CONSTANTS)])

    def __init__(self, args) -> None:
        super().__init__(args)

        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        train_dataset = SplitCORe50(base_path(), train=True,
                                    download=True, transform=transform, mini=False)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = avl_datasets.CORe50Dataset(base_path(), train=False,
                                                      download=True, transform=test_transform, mini=False)

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
            [transforms.ToPILImage(), SplitEmbeddedCORe50.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return MNISTMLP(512,
                        SplitEmbeddedCORe50.N_CLASSES_PER_TASK * SplitEmbeddedCORe50.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(*SplitEmbeddedCORe50.NORMALIZATION_CONSTANTS)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(*SplitEmbeddedCORe50.NORMALIZATION_CONSTANTS)

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SplitEmbeddedCORe50.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(
        ), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

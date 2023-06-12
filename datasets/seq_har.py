from argparse import Namespace
import os
import typing as t
from random import shuffle

import numpy as np
import torch
from scipy.io import loadmat
from torch import Tensor
from torch.utils import data
from torch.utils.data import Dataset
from backbone.HAR import DSADS_mlp, PAMAP2_mlp

from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset
import torch.nn.functional as F


def _partition(lst: t.List[t.Any], size: int) -> t.List[t.List[t.Any]]:
    """Partition a list into chunks of size `size`.
    :param lst: A list to partition.
    :param size: The size of each chunk.
    :yield: A generator of chunks.
    """
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _get_indices(targets: t.List[int]) -> t.Dict[int, t.List[int]]:
    """Get the indices of each class in a list of targets.

    :param targets: A list of targets.
    :return: A dictionary mapping each class to a list of indices.
    """
    indices: t.Dict[int, t.List[int]] = {}
    for i, target in enumerate(targets):
        target = int(target)
        if target not in indices:
            indices[target] = []
        indices[target].append(i)
    return indices


def _random_task_classes(
    task_count: int, class_count: int, drop_remainder=False
) -> t.List[t.List[int]]:
    """Assign each class to a random task.

    :param task_count: The number of tasks.
    :param class_count: The number of classes.
    :return: A dictionary mapping each task to a list of classes.
    """
    if class_count % task_count != 0 and not drop_remainder:
        raise ValueError("The classes need to be equally divided among tasks.")
    class_order = list(range(class_count))
    shuffle(class_order)
    if drop_remainder and class_count % task_count != 0:
        class_order = class_order[: -(class_count % task_count)]
    return _partition(class_order, class_count // task_count)


def _split_dataset_by_class(
    dataset: data.Dataset, targets: t.List[int]
) -> t.List[data.Dataset]:
    """Split a dataset into a dataset for each class.

    :param dataset: The dataset to split.
    :param targets: A list of targets or labels.
    :return: A list of datasets, one for each class.
    """
    indices = _get_indices(targets)
    return [data.Subset(dataset, indices[i]) for i in range(len(indices))]


class DSADS(Dataset):
    def __init__(self, root: str, train=False) -> None:
        super().__init__()
        self.root = root
        self.train = train
        self.data_filename = os.path.join(self.root, "HAR", "DSADS", "dsads.mat")

        self.class_names = [
            "ascending stairs",
            "cycling on an exercise bike in horizontal positions",
            "cycling on an exercise bike in vertical positions",
            "descending stairs",
            "exercising on a cross trainer",
            "exercising on a stepper",
            "jumping",
            "lying on back side",
            "lying on right side",
            "moving around in an elevator",
            "playing basketball",
            "rowing",
            "running on a treadmill3",
            "sitting",
            "standing",
            "standing in an elevator still",
            "walking in a parking lot",
            "walking on a treadmill1",
            "walking on a treadmill2",
        ]
        self.n_classes = len(self.class_names)

        """
        dsads.mat: 9120 * 408.
        Columns 0->404 are features, listed in the order of 'Torso', 'Right Arm',
        'Left Arm', 'Right Leg', and 'Left Leg'. Each position contains 81
        columns of features.
        Columns 405->407 are labels.
        Column 405 is the activity sequence indicating the executing of
        activities (usually not used in experiments).
        Column 406 is the activity label (1~19).
        Column 407 denotes the person (1~8).
        """

        mat = loadmat(os.path.join(root, "HAR", "DSADS", "dsads.mat"))
        np_data = mat["data_dsads"]
        # Remove column 405->407 because they are label information
        label = np_data[:, 406] - 1.0
        self.participants = np_data[:, 407].astype(int)
        np_data = np.delete(np_data, [405, 406, 407], axis=1)

        self.data = torch.from_numpy(np_data).float()
        self.targets = list(map(int, label))
        self.memory_usage_bytes = (
            self.data.element_size() * self.data.nelement() + 4 * len(self.targets)
        )

        # Remove participants 7 and 8 from the training set so they can be used
        # for testing. The choice of participants is arbitrary in this case.
        test_mask = np.isin(self.participants, [7, 8])
        if train:
            self.data = self.data[~test_mask]
            self.targets = np.array(self.targets)[~test_mask]
            self.participants = self.participants[~test_mask]
        else:
            self.data = self.data[test_mask]
            self.targets = np.array(self.targets)[test_mask]
            self.participants = self.participants[test_mask]

    def __getitem__(self, index: int) -> t.Tuple[Tensor, Tensor]:
        if self.train:
            # This is a hack to make the dataset compatible with mammoth
            # datasets which expect an un-augmented version of the data
            # to be returned as the third element of the tuple.
            return self.data[index], self.targets[index], self.data[index]
        else:
            return self.data[index], self.targets[index]


class PAMAP2(Dataset):
    def __init__(self, root: str, train=True) -> None:
        super().__init__()
        self.train = train

        self.root = root
        self.data_filename = os.path.join(root, "HAR", "PAMAP2", "pamap.mat")

        mat = loadmat(self.data_filename)["data_pamap"]

        # Columns 1~243 are features
        self.data = torch.tensor(mat[:, :243]).float()
        self.targets = torch.tensor(mat[:, 243] - 1.0).long()
        self.person = torch.tensor(mat[:, 244].astype(int))

        # Some labels are missing, so we remap the remaining labels to 0~11
        remapping = {
            11: 7,
            12: 8,
            15: 9,
            16: 10,
            23: 11,
        }
        for k, v in remapping.items():
            self.targets[self.targets == k] = v

        # Remap the labels to 0~11
        self.n_classes = 12

        # Remove participant 7,8 from the training set so they can be used
        # for testing, because these participants have the most measurements
        # in the most activities.
        mask = torch.isin(self.person, torch.tensor([7, 8]))
        if train:
            mask = ~mask

        self.data = self.data[mask]
        self.targets = self.targets[mask].tolist()
        self.person = self.person[mask]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> t.Tuple[Tensor, Tensor]:
        if self.train:
            # This is a hack to make the dataset compatible with mammoth
            # datasets which expect an un-augmented version of the data
            # to be returned as the third element of the tuple.
            return self.data[index], self.targets[index], self.data[index]
        else:
            return self.data[index], self.targets[index]


def split_HAR(
    train_dataset: t.Union[DSADS, PAMAP2],
    test_dataset: t.Union[DSADS, PAMAP2],
    task_count: int,
    fixed_class_order: t.Optional[t.List[int]] = None,
) -> t.Tuple[t.List[data.Dataset], t.List[data.Dataset]]:
    """
    Split DSADS into a continual learning training and testing set.
    """
    if type(train_dataset) != type(test_dataset):
        raise ValueError("Train and test datasets must be of the same type.")

    train_class_dataset = _split_dataset_by_class(train_dataset, train_dataset.targets)
    test_class_dataset = _split_dataset_by_class(test_dataset, test_dataset.targets)

    if fixed_class_order is None:
        task_composition = _random_task_classes(
            task_count, train_dataset.n_classes, drop_remainder=True
        )
    else:
        task_composition = _partition(
            fixed_class_order, len(fixed_class_order) // task_count
        )
        # Drop the remainder
        task_composition = task_composition[:task_count]

    train_task_datasets = []
    test_task_datasets = []
    for classes in task_composition:
        train_task_datasets.append(
            data.ConcatDataset([train_class_dataset[i] for i in classes])
        )
        test_task_datasets.append(
            data.ConcatDataset([test_class_dataset[i] for i in classes])
        )
    return train_task_datasets, test_task_datasets


class SequentialPAMAP(ContinualDataset):
    NAME = "seq-pamap"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 2
    N_TASKS = 6
    TRANSFORM = None

    def get_data_loaders(self):
        train_dataset = PAMAP2(base_path_dataset(), train=True)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, None, self.NAME)
        else:
            test_dataset = PAMAP2(base_path_dataset(), train=False)

        for dataset in [train_dataset, test_dataset]:
            # Perform a substitution to shuffle the class order
            dataset.targets = np.array(
                [self.substitution_table[y] for y in dataset.targets]
            )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return PAMAP2_mlp()

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size():
        return 500

    @staticmethod
    def get_minibatch_size():
        return SequentialPAMAP.get_batch_size()


class SequentialDSADS(ContinualDataset):
    NAME = "seq-dsads"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 2
    N_TASKS = 9
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.substitution_table = np.arange(19, dtype=int)
        if self.args.shuffle_classes:
            np.random.shuffle(self.substitution_table)

    def get_data_loaders(self):
        train_dataset = DSADS(base_path_dataset(), train=True)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, None, self.NAME)
        else:
            test_dataset = DSADS(base_path_dataset(), train=False)

        # Remove the last class because DSADS has 19 classes and we want 18
        # classes so that we can have 9 tasks with 2 classes each.
        for dataset in [train_dataset, test_dataset]:
            # Perform a substitution to shuffle the class order
            dataset.targets = np.array(
                [self.substitution_table[y] for y in dataset.targets]
            )
            dataset.data = dataset.data[dataset.targets != 18]
            dataset.targets = dataset.targets[dataset.targets != 18]

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return DSADS_mlp()

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size():
        return 500

    @staticmethod
    def get_minibatch_size():
        return SequentialPAMAP.get_batch_size()

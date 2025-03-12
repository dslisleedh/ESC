import math
import torch
from torch.utils.data.sampler import Sampler

import numpy as np


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class EnlargedRandomScaleSampler(Sampler):
    def __init__(self, dataset, batch_size_per_gpu, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.batch_size_per_gpu = batch_size_per_gpu
        self.scales = dataset.random_scales
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]
        
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        # scales for each batch
        n_iter = self.num_samples // self.batch_size_per_gpu
        n_res = self.num_samples % self.batch_size_per_gpu
        scales = np.random.choice(self.scales, n_iter, replace=True).tolist()
        scales = [scale for scale in scales for _ in range(self.batch_size_per_gpu)]
        scales += [1557] * n_res   # raise error if something goes wrong
        assert len(scales) == len(indices)
        assert len(indices) == self.num_samples
        
        return iter(zip(indices, scales))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
import numpy as np
import torch
import torch.utils.data


class StationaryMultivarGaussianDataset(torch.utils.data.IterableDataset):
    def __init__(self, N=100, D=10, mean=0, scale=1, transform=None, noise_scale=0):
        super(StationaryMultivarGaussianDataset).__init__()
        self.N = N
        self.D = D

        self.mean = torch.zeros(D) + mean
        self.scale = torch.zeros(D) + torch.Tensor(scale)

        self.transform = transform
        if self.transform is None:
            self.transform = torch.eye(D)
        self.transform = torch.Tensor(self.transform)

        self.noise_scale = noise_scale

    def __iter__(self):
        for i in range(self.N):
            sample = torch.mv(self.transform, torch.normal(0, self.scale)) \
                    + self.mean \
                    + torch.normal(torch.zeros(self.D), self.noise_scale) 
            yield i, sample

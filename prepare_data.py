import os
import random
from pathlib import Path

from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from visualize import *

pi = np.pi
e = np.e


class ToyDataset(Dataset):
    """
    Class providing simple toy data sets for synthetic experiments.
    The currently implemented data sets are listed & described below.
    Typically, the samples are uniformly or normally distributed on the described manifold,
    and observations are noisy (Gaussian noise with variance "noise").

    - 1D data set
        - "1Dgauss": simple Gaussian in 1D

    - 2D data sets
        - "eight": 1D manifold of lying eight (-> lemniscate of Gerono), uniform
        - "line": 1D line with slope & offset
        - "cosine": 1D manifold, cosine shaped
        - "circle": uniform samples from unit circle in 2D
        - "8gauss": Mixture of Gaussian in circular arrangement.

    - 3D data sets
        - "swiss_roll": 2D manifold from sklearn
        - "spiral_3D": 1D manifold

    - arbitrary dimension
        - "manifold": k-dimensional manifold in D-dimensional space:
            k dimension are i.i.d. drawn from normal/uniform distribution,
            remaining D-k dimensions are deterministic functions of (subset) of first k dimensions;
            k, D, sampling technique and functional dependencies can be changes in subroutine.
    """

    def __init__(self, config, n=2048, noise=0.05):
        """
        :param n: number of samples
        :param noise: level of (Gaussian) noise
        :param config: AttrDict config file
        """
        # 1D
        if config.dataset_type == "1Dgauss":
            self.samples = noise * torch.randn([n, 1]) + 3

        # 2D
        elif config.dataset_type == "eight":
            # lemniscate of Gerono (aka figure eight)
            t = 5 / 2 * pi * torch.rand([n, 1]) - 1 / 2 * pi
            samples = torch.cat((torch.cos(t), torch.sin(t) * torch.cos(t)), dim=1).float() / 2
            self.samples = samples + noise * torch.randn_like(samples)

        elif config.dataset_type == "line":
            t = 3 * torch.rand([n, 1])
            samples = torch.cat((t, 1.5 * t + 2), dim=1).float()
            self.samples = samples + noise * torch.randn_like(samples)

        elif config.dataset_type == "cosine":
            # cosine
            t = 3 * pi * torch.rand([n, 1])
            samples = 1 * torch.cat((t, torch.cos(t) + 1), dim=1).float()
            self.samples = samples + noise * torch.randn_like(samples)

        elif config.dataset_type == "circle":
            angle = 2 * pi * torch.rand([n, 1])
            radius = noise * torch.randn([n, 1]) + 1
            self.samples = torch.cat((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1).float()

        elif config.dataset_type == "8gauss":
            r = 0.4  # radius on which the centers will be arranged
            rs = np.sqrt(r ** 2 / 2)
            X, _ = make_blobs(n_samples=n,
                              centers=[[0, r], [0, -r], [r, 0], [-r, 0], [rs, rs], [-rs, -rs], [-rs, rs], [rs, -rs]],
                              cluster_std=noise, n_features=2)
            self.samples = torch.tensor(X).float() + 1

        # 3D
        elif config.dataset_type == "swiss_roll":
            X, _ = make_swiss_roll(n, noise=noise)

            self.samples = torch.tensor(X).float()

        elif config.dataset_type == "spiral_3D":
            k = 1.0  # number of maxima in [0,1]
            scale = 2.5
            t = torch.rand([n, 1])
            x = scale * torch.sin(2 * pi * k * t)  # * torch.cos(pi*t)
            y = scale * torch.cos(2 * pi * k * t)  # * torch.cos(pi*t)
            z = 1.2 * scale * k * t
            # rotate by 45 degree
            samples = torch.cat((x - z, y, z + x), dim=1).float()
            self.samples = samples + noise * torch.randn_like(samples) + 1.5

        # arbitrary dimension
        elif config.dataset_type == "manifold":

            # hyperparameter for synthetic dataset
            k = config.manifold_dims  # dimension of base distribution (and manifold)
            D = config.data_dims  # dimension of sample space X

            scale = 0.35  # scale parameter

            # base sampling in k dimensions
            k_noise = scale * torch.randn(n, k) - scale / 2

            if k < D:
                # add linear combination of random dimensions
                remaining_dims = torch.zeros(n, D - k)
                for i in range(D - k):
                    # n_combi = np.random.randint(low=1, high=min(5, k))  # how many initialized features to select
                    n_combi = 2
                    coefficients = torch.ones(2)
                    # coefficients = torch.rand(size=[n_combi]) - 1 / 2  # draw n_combi coefficients
                    # randomly select n_combi initialized features
                    k_choose = torch.randint(low=0, high=k, size=[n_combi])

                    for j in range(n_combi):
                        # linear combinations (all samples, but online the selected features
                        remaining_dims[:, i] += coefficients[j] * k_noise[:, k_choose[j]] ** 2

                    # apply some non-linear function here, e.g.g, bell curve exp(-x^2), cos(), etc
                    remaining_dims[:, i] = scale * torch.cos(remaining_dims[:, i])

                data = torch.cat((k_noise, remaining_dims), dim=1)

            self.samples = data  # + noise * torch.randn_like(data)

            # for visualization only:
            # data = pd.DataFrame(data)
            # fig = px.scatter_3d(data, x=0, y=1, z=2)
            # fig.show()

        else:
            raise AssertionError(f"Dataset ''{config.dataset_type}'' not implemented.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DatasetCELEBA(Dataset):
    def __init__(self, img_dir, n, transform):
        self.img_dir = img_dir
        self.transform = transform
        if n < 202599:
            self.imgs = random.sample(os.listdir(self.img_dir), k=n)
        else:
            self.imgs = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

    def get_test_val_split(self, return_split):
        np.random.seed(42)  # de-randomized

        dataset_size = len(self.imgs)
        indices = list(range(dataset_size))
        split = int(np.floor(0.33 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        if return_split == "train":
            return train_indices
        elif return_split == "val":
            return val_indices
        else:
            raise ValueError(f"Undefined split type '{return_split}'.")


class DatasetSUSY(Dataset):
    """
    Data set class providing SUSY data.
    """
    def __init__(self, csv_path, n_samples):
        data_file = pd.read_csv(csv_path, header=None)
        x = data_file.iloc[0:n_samples, 1:19].values  # read in features (18 features in total)
        y = data_file.iloc[0:n_samples, 0]  # read in label (not used)

        # sc = StandardScaler()
        sc = MinMaxScaler(feature_range=(0, 1))
        x = sc.fit_transform(x)

        # cast to tensor of floats
        self.x = torch.tensor(x).float()
        # self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def get_dataloader(config, test_shuffle=False):
    """
    Subroutine returning the data loaders as specified in the config
    :param test_shuffle: shuffling of test data (Bool, default: False)
    :param config: config dict, including "dataset_type", "epochs", etc.
    :return: train_dataloader, test_dataloader
    """
    if config.dataset_type == "MNIST":
        train_dataloader = DataLoader(
            datasets.MNIST(config.data_dir, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=config.batch_size, drop_last=True, shuffle=True)
        test_dataloader = DataLoader(
            datasets.MNIST(config.data_dir, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=config.batch_size, drop_last=True, shuffle=True)

    elif config.dataset_type == "CELEBA":
        N_SAMPLES = 202599  # 202.599 samples in total
        TEST_SIZE = 0.4

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(config.img_dim),
                                        transforms.ToTensor()])

        dataset = DatasetCELEBA(config.data_dir, N_SAMPLES, transform)

        # use random train & test splits (not last 500k as recommended for classification)
        train_indices, test_indices = train_test_split(range(N_SAMPLES), test_size=TEST_SIZE)
        train_split = Subset(dataset, train_indices)
        test_split = Subset(dataset, test_indices)

        train_dataloader = DataLoader(train_split, drop_last=True, num_workers=12, batch_size=config.batch_size)
        test_dataloader = DataLoader(test_split, shuffle=test_shuffle, drop_last=True, num_workers=12,
                                     batch_size=config.batch_size)

    elif config.dataset_type == "SUSY":
        csv_path = config.data_dir

        N_SAMPLES = 2500000  # 5.000.000 samples in total
        TEST_SIZE = 0.5

        dataset = DatasetSUSY(csv_path, n_samples=N_SAMPLES)

        train_indices, test_indices = train_test_split(range(N_SAMPLES), test_size=TEST_SIZE)
        train_split = Subset(dataset, train_indices)
        test_split = Subset(dataset, test_indices)

        train_dataloader = DataLoader(train_split, shuffle=True, drop_last=True, num_workers=12,
                                      batch_size=config.batch_size)
        test_dataloader = DataLoader(test_split, shuffle=True, drop_last=True, num_workers=12,
                                     batch_size=config.batch_size)
    else:
        path_to_train = Path(config.logdir + "/manifold_data_train.csv")
        path_to_test = Path(config.logdir + "/manifold_data_train.csv")
        if path_to_train.is_file():
            # assuming that test data is also saved if training data exists
            train_split = pd.read_csv(path_to_train, header=None)
            test_split = pd.read_csv(path_to_test, header=None)
            # convert to torch tensors
            train_split = torch.tensor(train_split.values).float()
            test_split = torch.tensor(test_split.values).float()
        else:
            N_SAMPLES = 50000
            TEST_SIZE = 0.5

            dataset = ToyDataset(config, n=N_SAMPLES)

            sc = MinMaxScaler(feature_range=(0., 1.))
            dataset = torch.tensor(sc.fit_transform(dataset.samples)).float()

            train_indices, test_indices = train_test_split(range(N_SAMPLES), test_size=TEST_SIZE)
            train_split = Subset(dataset, train_indices)
            test_split = Subset(dataset, test_indices)

            # save the data sets for post-processing
            manifold_data_train = train_split.dataset.numpy()
            manifold_data_train = pd.DataFrame(manifold_data_train)
            manifold_data_train.to_csv(config.logdir + '/manifold_data_train.csv', index=False, header=False)

            manifold_data_test = test_split.dataset.numpy()
            manifold_data_test = pd.DataFrame(manifold_data_test)
            manifold_data_test.to_csv(config.logdir + '/manifold_data_test.csv', index=False, header=False)

        train_dataloader = DataLoader(train_split, batch_size=config.batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_split, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # set data dimensionality
    if config.dataset_type == "MNIST":
        dims = train_dataloader.dataset[0][0].size()
        config.data_dims = dims[0] * dims[1] * dims[2]
    elif config.dataset_type == "CELEBA":
        dims = train_dataloader.dataset[0].size()
        config.data_dims = dims[0] * dims[1] * dims[2]
    else:
        config.data_dims = [*train_dataloader.dataset[0].size()][0]

    # plot target data (if dim <= 3)
    if config.data_dims == 2:
        plot_2D(train_dataloader.dataset, 'Target distribution')
    if config.data_dims == 3:
        plot_3D(train_dataloader.dataset, 'Target distribution')

    return train_dataloader, test_dataloader

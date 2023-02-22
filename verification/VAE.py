# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.


import os, sys
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torchvision

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class LinearVAE(Module):

    def __init__(self, latent_dim, output_dim, num_samples=100):
        super(LinearVAE, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_samples = num_samples

        self.encoder = Linear(output_dim, latent_dim)
        self.decoder = Linear(latent_dim, output_dim)
        self.noise_std = Parameter(torch.Tensor([0.]))
        self.q_z_std = Parameter(torch.Tensor(np.zeros(latent_dim)))
        self.p_z = Normal(torch.Tensor([0.]), torch.Tensor([1.]))

    def forward(self, x):
        N, D = x.shape

        z_params = self.encoder(x)
        tau = softplus(self.q_z_std)
        q_z = Normal(z_params, tau)

        z_samples = q_z.rsample([self.num_samples])
        p_x_z_mean = self.decoder(z_samples)
        sigma = softplus(self.noise_std)
        p_x_z = Normal(p_x_z_mean, sigma)

        three_entropies_linear = self.three_entropies_linear(N, D, sigma, tau)
        lower_bound = p_x_z.log_prob(x).mean(0).sum() - kl_divergence(q_z, self.p_z).sum()

        return lower_bound, three_entropies_linear

    @staticmethod
    def three_entropies_linear(N, D, sigma, tau):
        """Three Entropies (accumulated) for linear VAE, Eq.(37).
        
        Parameters
        ----------
            N (int) : batch size
            D (int) : output dimensionality
            sigma (torch.tensor, size=(1,)) : decoder standard deviation
            tau (torch.tensor, size=(H,)    : encoder standard deviation        
        """
        return N * (-D / 2 * (torch.log(2 * torch.Tensor([np.pi])) + 1) - D * torch.log(sigma) + torch.log(tau).sum())

    def compute_log_likelihood(self, data_loader):
        return compute_log_likelihood(self.decoder.weight, self.decoder.bias, softplus(self.noise_std), data_loader)


class VAE1(Module):

    def __init__(self, latent_dim, output_dim, encoder_hidden_units, decoder_hidden_units, num_samples=100):
        super(VAE1, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_samples = num_samples

        self.encoder = self.construct_dnn([output_dim] + encoder_hidden_units + [latent_dim * 2])
        self.decoder = self.construct_dnn([latent_dim] + encoder_hidden_units + [output_dim])
        self.noise_std = Parameter(torch.Tensor([0.]))
        self.p_z = Normal(torch.Tensor([0.]), torch.Tensor([1.]))

    def forward(self, x):
        N, D = x.shape

        z_params = self.encoder(x)
        tau = softplus(z_params[:, self.latent_dim:])
        q_z = Normal(z_params[:, :self.latent_dim], tau)

        z_samples = q_z.rsample([self.num_samples])
        p_x_z_mean = self.decoder(z_samples)
        sigma = softplus(self.noise_std)
        p_x_z = Normal(p_x_z_mean, sigma)

        three_entropies = self.three_entropies_nonlinear(N, D, sigma, tau)
        lower_bound = p_x_z.log_prob(x).mean(0).sum() - kl_divergence(q_z, self.p_z).sum()

        return lower_bound, three_entropies

    @staticmethod
    def three_entropies_nonlinear(N, D, sigma, tau):
        """Three Entropies (accumulated) for non-linear VAE (VAE-1), Eq.(7).
        
        Parameters
        ----------
            N (int) : batch size
            D (int) : output dimensionality
            sigma (torch.tensor, size=(1,)) : decoder standard deviation
            tau (torch.tensor, size=(N, H)) : encoder standard deviation
        """
        return N * D * (-1 / 2 * (torch.log(2 * torch.Tensor([np.pi])) + 1) - torch.log(sigma)) + torch.log(tau).sum()

    @staticmethod
    def construct_dnn(units):
        layers = []
        for i in range(len(units) - 1):
            layers.append(Linear(units[i], units[i + 1]))
            if i < len(units) - 2:
                layers.append(ReLU())
        return Sequential(*layers)


class VAE3(Module):

    def __init__(self, latent_dim, output_dim, encoder_hidden_units, decoder_hidden_units, num_samples=100):
        super(VAE3, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_samples = num_samples

        self.encoder = self.construct_dnn([output_dim] + encoder_hidden_units + [latent_dim * 2])
        self.decoder = self.construct_dnn([latent_dim] + encoder_hidden_units + [output_dim * 2])
        self.p_z = Normal(torch.Tensor([0.]), torch.Tensor([1.]))

    def forward(self, x):
        N, D = x.shape

        z_params = self.encoder(x)
        tau = softplus(z_params[:, self.latent_dim:])
        q_z = Normal(z_params[:, :self.latent_dim], tau)
        z_samples = q_z.rsample([self.num_samples])

        tmp = self.decoder(z_samples)
        p_x_z_mean = tmp[:, :, :self.output_dim]
        sigma = softplus(tmp[:, :, self.output_dim:])
        p_x_z = Normal(p_x_z_mean, sigma)

        three_entropies = self.three_entropies_sigmaz(N, D, sigma, tau)
        lower_bound = p_x_z.log_prob(x).mean(0).sum() - kl_divergence(q_z, self.p_z).sum()

        return lower_bound, three_entropies

    @staticmethod
    def three_entropies_sigmaz(N, D, sigma, tau):
        """Three Entropies (accumulated) for sigma(z)-VAE (VAE-3), Eq.(31).
        Args:
            N (int) : batch size
            D (int) : output dimensionality
            sigma (torch.tensor, size=(num_samples, N, D)) : decoder std
            tau (torch.tensor, size=(N, H)) : encoder standard deviation
        """
        return -N * D / 2 * (torch.log(2 * torch.Tensor([np.pi])) + 1) - torch.log(sigma).mean(0).sum() \
               + torch.log(tau).sum()

    @staticmethod
    def construct_dnn(units):
        layers = []
        for i in range(len(units) - 1):
            layers.append(Linear(units[i], units[i + 1]))
            if i < len(units) - 2:
                layers.append(ReLU())
        return Sequential(*layers)


def compute_log_likelihood(weight, bias, sigma, data_loader):
    output_dim = bias.shape[0]

    data_size = 0
    S = None
    for x, _ in data_loader:
        x = x.cuda()
        x = x.reshape(x.shape[0], -1)
        tmp = x - bias
        if S is None:
            S = torch.matmul(tmp.T, tmp)
        else:
            S = S + torch.matmul(tmp.T, tmp)
        data_size += x.shape[0]

    S = S / data_size
    C = torch.matmul(weight, weight.T) + \
        torch.eye(output_dim) * sigma ** 2
    L_C = torch.cholesky(C)
    CinvS = torch.triangular_solve(torch.triangular_solve(S, L_C, upper=False)[0].T, L_C, upper=False)[0]
    log_likelihood = (-(output_dim) / 2 * np.log(2 * np.pi) - torch.log(torch.diag(L_C)).sum() \
                      - torch.diag(CinvS).sum() / 2.) * data_size

    return log_likelihood


def train_vae(vae, num_iters, train_loader, test_loader, lr=0.1, compute_ll=False):
    opt = optim.Adam(vae.parameters(), lr=lr)

    bounds = []
    three_entropies = []
    log_likelihood = []
    test_log_likelihood = []

    for epoch in tqdm(range(num_iters)):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            opt.zero_grad()
            bound = vae(data.reshape(data.shape[0], -1))[0]
            (-bound).backward()
            opt.step()

        three_entropies_iteration = 0
        bound_iteration = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            with torch.no_grad():
                bound_batch, three_entropies_batch = vae(data.reshape(data.shape[0], -1))
                bound_iteration += float(bound_batch)
                three_entropies_iteration += float(three_entropies_batch)
        bounds.append(float(bound_iteration))
        three_entropies.append(float(three_entropies_iteration))
        if compute_ll:
            with torch.no_grad():
                logL = vae.compute_log_likelihood(train_loader)
                test_logL = vae.compute_log_likelihood(test_loader)
            log_likelihood.append(float(logL))
            test_log_likelihood.append(float(test_logL))
        else:
            log_likelihood = None
            test_log_likelihood = None

    return bounds, three_entropies, log_likelihood, test_log_likelihood


def generate_PCA_data(num_data_train, num_data_test, latent_dim, output_dim, batch_size, seed=None):
    """Generate PCA data.

        Args:
            num_data_train (int): number of training data points to generate
            num_data_test (int): number of test data points to generate
            latent_dim (int): H - latent dimensionality
            output_dim (int): D - output dimensionality
            batch_size (int)
            seed (int): random seed
    """

    if seed is None:
        seed = np.random.randint(2 ** 32 - 1)

    np.random.seed(seed)

    num_data = num_data_train + num_data_test

    W_gen = np.random.uniform(low=0, high=1, size=(output_dim, latent_dim))
    mu_gen = np.random.uniform(low=0, high=1, size=output_dim)
    sigma_gen = 0.1

    z = np.random.normal(0, 1, (num_data, latent_dim))
    x = np.einsum('dh,nh->nd', W_gen, z) + mu_gen + np.random.normal(0, sigma_gen, (num_data, output_dim))

    data_train = torch.Tensor(x[:num_data_train])
    data_test = torch.Tensor(x[num_data_train:])
    dummy_labels_train = torch.zeros(num_data_train, dtype=torch.int32)
    dummy_labels_test = torch.zeros(num_data_test, dtype=torch.int32)

    dataset_train = TensorDataset(data_train, dummy_labels_train)
    dataset_test = TensorDataset(data_test, dummy_labels_test)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True)

    W_gen = torch.Tensor(W_gen).cuda()
    mu_gen = torch.Tensor(mu_gen).cuda()
    sigma_gen = torch.Tensor([sigma_gen]).cuda()
    ground_truth_ll = compute_log_likelihood(W_gen, mu_gen, sigma_gen, train_loader)
    ground_truth_ll = np.asarray([ground_truth_ll.cpu()])

    return train_loader, test_loader, ground_truth_ll


def generate_PCA_ring_data(num_data_train, num_data_test, latent_dim, output_dim, batch_size, seed=None):
    """Generate PCA data.

        Args:
            num_data_train (int): number of training data points to generate
            num_data_test (int): number of test data points to generate
            latent_dim (int): H - latent dimensionality
            output_dim (int): D - output dimensionality
            seed (int): random seed
    """

    if seed is None:
        seed = np.random.randint(2 ** 32 - 1)

    np.random.seed(seed)

    num_data = num_data_train + num_data_test

    W_gen = np.random.uniform(low=0, high=1, size=(output_dim, latent_dim))
    mu_gen = np.random.uniform(low=0, high=1, size=output_dim)
    sigma_gen = 0.1

    z = np.random.normal(0, 1, (num_data, latent_dim))
    z = z / 10 + z / np.linalg.norm(z, axis=1, keepdims=True)
    x = np.einsum('dh,nh->nd', W_gen, z) + mu_gen + np.random.normal(0, sigma_gen, (num_data, output_dim))

    data_train = torch.Tensor(x[:num_data_train])
    data_test = torch.Tensor(x[num_data_train:])
    dummy_labels_train = torch.zeros(num_data_train, dtype=torch.int32)
    dummy_labels_test = torch.zeros(num_data_test, dtype=torch.int32)

    dataset_train = TensorDataset(data_train, dummy_labels_train)
    dataset_test = TensorDataset(data_test, dummy_labels_test)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True)

    W_gen = torch.Tensor(W_gen).cuda()
    mu_gen = torch.Tensor(mu_gen).cuda()
    sigma_gen = torch.Tensor([sigma_gen]).cuda()
    ground_truth_ll = compute_log_likelihood(W_gen, mu_gen, sigma_gen, train_loader)
    ground_truth_ll = np.asarray([ground_truth_ll.cpu()])

    return train_loader, test_loader, ground_truth_ll


def visualize(bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll, dataset, model):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle(f'{model} VAE ({dataset} data)')

    for ax in (ax1, ax2):
        niter = range(1, len(three_entropies) + 1)
        if log_likelihood is not None:
            ax.plot(niter, log_likelihood, label='loglikelihood', color='tab:red', zorder=2)
        ax.plot(niter, bounds, label='lower bound', color='lightblue', zorder=1)
        if test_log_likelihood is not None:
            ax.plot(niter, test_log_likelihood, '--', label='held-out loglikelihood', color='tab:orange', zorder=3)
        ax.plot(niter, three_entropies, label='three entropies', color='tab:green', zorder=4)
        if ground_truth_ll is not None:
            ax.axhline(ground_truth_ll, linestyle='--', color='black', label='groud-truth loglikelihood', zorder=5)
        ax.set_xlim(0, len(three_entropies))
        ax.grid(linestyle=':', linewidth=0.4)

    bottom = np.min([np.mean(v[len(v) // 3:]) for v in
                     [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if v is not None])
    top = np.max([v[-1] for v in [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if
                  v is not None])
    bottom -= np.abs(bottom - top) * 0.5
    top += np.abs(bottom - top) * 0.1

    ax2.set_ylim(bottom=bottom, top=top)
    ax2.set_xlabel('Iteration')

    handles, labels = ax1.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.legend(handles, labels, loc="lower center", ncol=2, prop={'size': 10});

    return fig


if __name__ == "__main__":

    print(f"PyTorch Version {torch.__version__}")
    print(torch.cuda.get_device_capability(torch.cuda.current_device()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    model = sys.argv[1]
    dataset = sys.argv[2]
    # seed = 0
    seed = np.random.randint(2 ** 32 - 1)

    # default parameters
    if dataset == 'PCA':
        params = {
            "ntrain": 10000,
            "ntest": 10000,
            "H": 2,
            "D": 10,
            "niters": 1500,
            "lr": 0.001,
            "nsamples": 100,
            "batch_size": 2000
        }
    elif dataset == 'MNIST':
        params = {
            "ntrain": 60000,
            "ntest": 10000,
            "H": 2,
            "D": 28 * 28,
            "niters": 200,
            "lr": 0.001,
            "nsamples": 100,
            "batch_size": 2000
        }
    elif dataset == 'PCA-ring':
        params = {
            "ntrain": 10000,
            "ntest": 10000,
            "H": 2,
            "D": 10,
            "niters": 3000,
            "lr": 0.001,
            "nsamples": 100,
            "batch_size": 2000
        }

    # parameters given by user via command line
    try:
        if len(sys.argv) > 3:
            user_params = dict(arg.split('=', 1) for arg in sys.argv[3:])
        else:
            user_params = {}
    except:
        print('WARNING: Could not read user parameters properly. Reverting to default parameters.')
        user_params = {}
    params.update(user_params)

    # convert strings to appropriate types
    num_data_train = int(params['ntrain'])
    num_data_test = int(params['ntest'])
    latent_dim = int(params['H'])
    output_dim = int(params['D'])
    num_iters = int(params['niters'])
    lr = float(params['lr'])
    nsamples = int(params['nsamples'])
    batch_size = int(params['batch_size'])

    print(f'{model} VAE on {dataset}')
    print(params)

    # create directories
    output_folder = f"./output/{model}/{dataset}/H={latent_dim}, D={output_dim}, ntrain={num_data_train}, ntest={num_data_test}, batch_size={batch_size}, lr={lr}, niters={num_iters}, nsamples={nsamples}"
    race = True
    while race:
        suffix = ''
        npath = 0
        try:
            while (os.path.exists(output_folder + suffix + '/')):
                suffix = ' (' + str(npath) + ')'
                npath += 1
            os.makedirs(output_folder + suffix + '/')
            race = False
        except OSError:
            pass
    output_folder += suffix + '/'
    print(f'Output to {output_folder}')

    # instantiate model
    torch.manual_seed(seed)
    if model == 'linear':
        vae = LinearVAE(latent_dim, output_dim, nsamples)
        compute_ll = True
    elif model == 'non-linear':
        encoder_hidden_units = [50, 50]
        decoder_hidden_units = [50, 50]
        vae = VAE1(latent_dim, output_dim, encoder_hidden_units, decoder_hidden_units, nsamples)
        compute_ll = False
    elif model == 'sigma-z-full':
        encoder_hidden_units = [50, 50]
        decoder_hidden_units = [50, 50]
        vae = VAE3(latent_dim, output_dim, encoder_hidden_units, decoder_hidden_units, nsamples)
        compute_ll = False
    vae = vae.cuda()

    # load data
    if dataset == 'PCA':
        train_loader, test_loader, ground_truth_ll = generate_PCA_data(num_data_train, num_data_test, latent_dim,
                                                                       output_dim, batch_size, seed)
    elif dataset == 'PCA-ring':
        train_loader, test_loader, ground_truth_ll = generate_PCA_ring_data(num_data_train, num_data_test, latent_dim,
                                                                            output_dim, batch_size, seed)
    elif dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                './data/', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()
                ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                './data/', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()
                ])),
            batch_size=batch_size, shuffle=True)
        ground_truth_ll = None

    # train model
    bounds, three_entropies, log_likelihood, test_log_likelihood = train_vae(vae, num_iters, train_loader, test_loader,
                                                                             lr, compute_ll)

    # save results
    torch.save(vae.state_dict(), output_folder + 'vae')
    np.savetxt(output_folder + 'bounds.txt', bounds)
    np.savetxt(output_folder + 'three_entropies.txt', three_entropies)
    if compute_ll:
        np.savetxt(output_folder + 'log_likelihood.txt', log_likelihood)
        np.savetxt(output_folder + 'test_log_likelihood.txt', test_log_likelihood)
    if ground_truth_ll is not None:
        np.savetxt(output_folder + 'ground_truth_ll.txt', ground_truth_ll)
    fig = visualize(bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll, dataset, model)
    fig.savefig(output_folder + 'plot.jpg')

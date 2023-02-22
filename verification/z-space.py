import os, sys
import gc
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torchvision

torch.set_default_tensor_type('torch.cuda.FloatTensor')

import importlib

VAE = importlib.import_module('VAE')

num_vis = 1000
num_train_vis = num_vis
seed = 0

if len(sys.argv) > 3:
    seed = int(sys.argv[3])


def nvis():
    i = 0
    while True:
        i += 1
        yield int(1 + 0.1 * i)


def train_vae(vae, num_iters, train_loader, test_loader, lr=0.1, compute_ll=False, ground_truth_ll=None, title='',
              num_data_train=10000, num_data_test=10000):
    opt = optim.Adam(vae.parameters(), lr=lr)

    bounds = []
    three_entropies = []
    log_likelihood = []
    test_log_likelihood = []
    z_list = []
    tau_list = []
    sigma_list = []
    all_data = torch.cat([data for data, _ in train_loader], dim=0)
    all_data = all_data.reshape(all_data.shape[0], -1)
    vis_data = all_data[:num_vis]
    n_vis_epoch = nvis()
    next_vis_epoch = next(n_vis_epoch)
    pca = PCA(2)
    pca.fit(all_data.cpu().numpy())
    train_pca = pca.transform(all_data.cpu().numpy())

    for epoch in tqdm(range(1, num_iters + 1)):
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

        with torch.no_grad():
            data = vis_data.cuda()
            z_params = vae.encoder(data.reshape(data.shape[0], -1))
            z = z_params[:, :vae.latent_dim]

            x = vae.decoder(z)[:, :vae.output_dim]
            if model == 'linear':
                tau = softplus(vae.q_z_std)
                tau_list.append([tau.cpu().numpy()])
                sigma = softplus(vae.noise_std).cpu().numpy()
                sigma_list.append([sigma])
            elif model == 'non-linear':
                tau = softplus(z_params[:, vae.latent_dim:])
                tau_list.append(tau.cpu().numpy())
                sigma = softplus(vae.noise_std).cpu().numpy()
                sigma_list.append([sigma])
            elif model == 'sigma-z-full':
                tau = softplus(z_params[:, vae.latent_dim:])
                tau_list.append(tau.cpu().numpy())
                sigma = softplus(vae.decoder(z)[:, vae.output_dim:]).cpu().numpy()
                sigma_list.append(sigma)

            tau_list = tau_list[-20:]
            sigma_list = sigma_list[-20:]
            z_list.append(z.cpu().numpy())
            z_list = z_list[-20:]

            if (epoch % next_vis_epoch == 0) or (epoch == num_iters):
                visualize_z_space(z_list, tau_list, sigma_list, epoch, bounds, three_entropies, log_likelihood,
                                  test_log_likelihood, ground_truth_ll, title, train_pca, x, pca, num_data_train,
                                  num_data_test)
                next_vis_epoch = next(n_vis_epoch)

    return bounds, three_entropies, log_likelihood, test_log_likelihood


def visualize_z_space(z_list, tau_list, sigma_list, epoch, bounds, three_entropies, log_likelihood, test_log_likelihood,
                      ground_truth_ll, title, train_pca, x, pca, num_data_train, num_data_test):
    z_list = torch.tensor(z_list, device='cpu')
    z = z_list[-1]
    x = x.cpu().numpy()
    tau_list = torch.tensor(tau_list, device='cpu')
    tau = tau_list[-1]
    sigma_list = torch.tensor(sigma_list, device='cpu')
    sigma = sigma_list[-1].cpu().numpy()
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{title} - epoch {epoch}', fontsize=24)

    gs = fig.add_gridspec(2, 3)
    gs.update(wspace=0.25, hspace=0.3)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title('z-space', fontsize=16)
    ax2 = fig.add_subplot(gs[:, 1])
    if x.shape[1] > 2:
        x_pca = pca.transform(x)
        ax2.set_title('x-space (PCA projection)', fontsize=16)
        ax2.set_xlabel('Principal Component 1', fontsize=10)
        ax2.set_ylabel('Principal Component 2', fontsize=10)
    else:
        x_pca = x
        ax2.set_title('x-space')
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])
    fig.subplots_adjust(left=0.06, right=0.94, top=0.83, bottom=0.16)

    ax1.scatter(z[:, 0], z[:, 1], s=1, label=r'$z(x_\mathrm{train})$', color='tab:red', zorder=2)
    ax1.grid(linestyle=':', linewidth=0.4)

    if model == 'linear':
        for (z_x, z_y) in z:
            ax1.add_patch(
                mpatches.Rectangle(xy=(z_x - tau[:, 0], z_y - tau[:, 1]), width=2 * tau[:, 0], height=2 * tau[:, 1],
                                   linewidth=1, color='tab:blue', fill=False, zorder=1))
    else:
        for ((z_x, z_y), (tau_x, tau_y)) in zip(z, tau):
            ax1.add_patch(
                mpatches.Rectangle(xy=(z_x - tau_x, z_y - tau_y), width=2 * tau_x, height=2 * tau_y, linewidth=1,
                                   color='tab:blue', fill=False, zorder=1))

    handles, labels = ax1.get_legend_handles_labels()
    blue_rectangle = mpatches.Rectangle((0.5, 0.5), 1, 1, color='tab:blue', fill=False)
    handles += [blue_rectangle]
    labels += [r"   $\tau_1 \times \tau_2$"]
    ax1.legend(handles, labels, loc="lower center", ncol=2, prop={'size': 10}, markerscale=3, handletextpad=0.)

    zpt = z_list + tau_list
    zmt = z_list - tau_list
    left1, bottom1 = zmt.min(dim=1)[0].mean(0) * 1.1
    right1, top1 = zpt.max(dim=1)[0].mean(0) * 1.1
    bottom1 -= 0.1 * (top1 - bottom1)
    dh = (top1 - bottom1)
    dw = (right1 - left1)
    if dh > dw:
        left1 -= (dh - dw) / 2.
        right1 += (dh - dw) / 2.
    elif dw > dh:
        bottom1 -= (dw - dh) / 2.
        top1 += (dw - dh) / 2.
    # print(epoch, bottom1, top1, left1, right1)
    ax1.set_ylim(bottom=bottom1, top=top1)
    ax1.set_xlim(left=left1, right=right1)

    ax2.scatter(train_pca[:num_train_vis, 0], train_pca[:num_train_vis, 1], s=1, label=r'$x_\mathrm{train}$',
                color='grey', zorder=1)
    ax2.scatter(x_pca[:, 0], x_pca[:, 1], s=1, label=r'$x(z)$', color='tab:red', zorder=2)
    ax2.grid(linestyle=':', linewidth=0.4)

    if model == 'linear' or model == 'non-linear':
        sigma = sigma[0]
        for x_i in x:
            if x.shape[1] > 2:
                x_pos, y_pos = pca.transform([x_i - sigma])[0]
                width, height = pca.transform([x_i + sigma])[0] - pca.transform([x_i - sigma])[0]
            else:
                x_pos, y_pos = x_i - sigma[0]
                width, height = 2 * sigma[0]
            ax2.add_patch(
                mpatches.Rectangle(xy=(x_pos, y_pos), width=width, height=height, linewidth=1, color='tab:blue',
                                   fill=False, zorder=1.5))
    else:
        for x_i, sigma_i in zip(x, sigma):
            if x.shape[1] > 2:
                x_pos, y_pos = pca.transform([x_i - sigma_i])[0]
                width, height = pca.transform([x_i + sigma_i])[0] - pca.transform([x_i - sigma_i])[0]
            else:
                x_pos, y_pos = x_i - sigma_i
                width, height = 2 * sigma_i
            ax2.add_patch(
                mpatches.Rectangle(xy=(x_pos, y_pos), width=width, height=height, linewidth=1, color='tab:blue',
                                   fill=False, zorder=1.5))

    handles, labels = ax2.get_legend_handles_labels()
    blue_rectangle = mpatches.Rectangle((0.5, 0.5), 1, 1, color='tab:blue', fill=False)
    handles += [blue_rectangle]
    labels += [r"   $\sigma_\mathrm{PC1} \times \sigma_\mathrm{PC2}$"]

    ax2.legend(handles, labels, loc="lower center", ncol=3, prop={'size': 10}, markerscale=3, handletextpad=0.);

    # ax2.set_ylim(-2, 2)
    # ax2.set_xlim(-2, 2)

    bounds = np.asarray(bounds) / num_data_train
    three_entropies = np.asarray(three_entropies) / num_data_train
    if log_likelihood is not None:
        log_likelihood = np.asarray(log_likelihood) / num_data_train
    if test_log_likelihood is not None:
        test_log_likelihood = np.asarray(test_log_likelihood) / num_data_test
    if ground_truth_ll is not None:
        ground_truth_ll = ground_truth_ll / num_data_train
    for ax in (ax3, ax4):
        niter = range(1, len(three_entropies) + 1)
        if log_likelihood is not None:
            ax.plot(niter, log_likelihood, label='loglikelihood', color='tab:red', zorder=2)
        if test_log_likelihood is not None:
            ax.plot(niter, test_log_likelihood, '--', label='held-out logl.', color='tab:orange', zorder=3)
        if ground_truth_ll is not None:
            ax.axhline(ground_truth_ll, linestyle=':', color='black', label='ground-truth', zorder=5)
        ax.plot(niter, bounds, label='lower bound', color='lightblue', zorder=1)
        ax.plot(niter, three_entropies, label='three entropies', color='tab:green', zorder=4)
        ax.set_xlim(0, len(three_entropies))
        ax.grid(linestyle=':', linewidth=0.4)

    bottom3 = np.min(
        [np.mean(v) for v in [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if
         v is not None])
    top3 = np.max([np.mean(v[max(0, int(len(v) * 0.9) - 100):]) for v in
                   [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if v is not None])
    max_std3 = np.max([np.std(v[max(0, int(len(v) * 0.9) - 100):]) for v in
                       [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if
                       v is not None])
    dh3 = max((top3 - bottom3), max_std3 * 50)
    bottom3 = (top3 + bottom3) / 2. - dh3 / 2.
    top3 = (top3 + bottom3) / 2. + dh3 / 2.
    bottom3 -= np.abs(bottom3 - top3) * 0.1
    top3 += np.abs(bottom3 - top3) * 0.1

    bottom4 = np.min([np.mean(v[len(v) // 2:]) for v in
                      [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if v is not None])
    top4 = np.max([np.mean(v[-min(len(v), 100):]) for v in
                   [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if v is not None])
    max_std4 = np.max([np.std(v[-min(len(v), 100):]) for v in
                       [bounds, three_entropies, log_likelihood, test_log_likelihood, ground_truth_ll] if
                       v is not None])
    dh4 = max((top4 - bottom4), max_std4 * 10)
    bottom4 = (top4 + bottom4) / 2. - dh4 / 2.
    top4 = (top4 + bottom4) / 2. + dh4 / 2.
    bottom4 -= np.abs(bottom4 - top4) * 0.2
    top4 += np.abs(bottom4 - top4) * 0.1

    ax3.set_ylim(bottom=bottom3, top=top3)
    ax4.set_ylim(bottom=bottom4, top=top4)
    ax4.set_xlim(max(0, len(bounds) - 100), int(len(bounds)))
    ax4.set_xlabel('Epoch', fontsize=10)

    # for ax in (ax1, ax2, ax3):
    #     ax.tick_params(length=6, width=2)    

    handles, labels = ax3.get_legend_handles_labels()
    if model == 'linear':
        ax4.legend(handles, labels, loc="lower right", ncol=2, prop={'size': 8});
    else:
        ax4.legend(handles, labels, loc="lower center", ncol=3, prop={'size': 8});

        # set_size(fig, (10, 5), eps=1e-5, give_up=100)
    directory = f'./output/z-space/{model.capitalize()}VAE_{dataset}_lr={lr}_bs={batch_size}_nvis={num_vis}_ntvis={num_train_vis}_seed={seed}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/{epoch}.jpg', dpi=100)
    if model == 'linear':
        nfig = 'Fig4a'
    if model == 'non-linear':
        nfig = 'Fig4b'
    if model == 'sigma-z-full':
        nfig = 'Fig4c'
    plt.savefig(f'{directory}/{nfig}_{epoch}.pdf', format='pdf', bbox_inches='tight', dpi=100)
    fig.clf()
    plt.close('all')
    gc.collect()


if __name__ == "__main__":

    model = sys.argv[1]
    dataset = sys.argv[2]

    # parameters
    if dataset == 'PCA' or dataset == 'PCA-ring':
        num_data_train = 10000
        num_data_test = 10000
        latent_dim = 2
        output_dim = 10

        lr = 0.001
        batch_size = 2000
        nsamples = 100
        num_iters = 2000

    elif dataset == 'MNIST':
        num_data_train = 60000
        num_data_test = 10000
        latent_dim = 2
        output_dim = 28 * 28

        lr = 0.001
        batch_size = 2000
        nsamples = 100
        num_iters = 5000

    if dataset == 'PCA-ring':
        num_iters = 3000

    # instantiate model
    torch.manual_seed(seed)
    if model == 'linear':
        vae = VAE.LinearVAE(latent_dim, output_dim, nsamples)
        compute_ll = True
        model_name = 'Linear VAE'
    elif model == 'non-linear':
        encoder_hidden_units = [50, 50]
        decoder_hidden_units = [50, 50]
        vae = VAE.VAE1(latent_dim, output_dim, encoder_hidden_units, decoder_hidden_units, nsamples)
        compute_ll = False
        model_name = 'VAE-1'
    elif model == 'sigma-z-full':
        encoder_hidden_units = [50, 50]
        decoder_hidden_units = [50, 50]
        vae = VAE.VAE3(latent_dim, output_dim, encoder_hidden_units, decoder_hidden_units, nsamples)
        compute_ll = False
        model_name = 'VAE-3'
    vae = vae.cuda()

    # load data
    if dataset == 'PCA':
        title = f'{model_name} on PCA data set'
        train_loader, test_loader, ground_truth_ll = VAE.generate_PCA_data(num_data_train, num_data_test, latent_dim,
                                                                           output_dim, batch_size, seed)
    elif dataset == 'PCA-ring':
        title = f'{model_name} on PCA-ring data set'
        train_loader, test_loader, ground_truth_ll = VAE.generate_PCA_ring_data(num_data_train, num_data_test,
                                                                                latent_dim, output_dim, batch_size,
                                                                                seed)
    elif dataset == 'MNIST':
        title = f'{model_name} on MNIST data set'
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
    if dataset != 'PCA':
        ground_truth_ll = None

    # train model
    plt.ioff()
    bounds, three_entropies, log_likelihood, test_log_likelihood = train_vae(vae, num_iters, train_loader, test_loader,
                                                                             lr, compute_ll, ground_truth_ll, title,
                                                                             num_data_train, num_data_test)

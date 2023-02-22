"""Helper functions"""
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
import numpy as np
import torch

import plotly.express as px
import pandas as pd
import plotly.io as pio
from matplotlib import cm

from utils import ema

pio.renderers.default = "browser"

from scipy.stats import multivariate_normal


# 2D plot
def plot_2D(data, title=None):
    fig = plt.figure()
    if title is not None:
        plt.title(f"{title}")
    plt.plot(data[:, 0], data[:, 1], marker='o', color='blue', alpha=0.4,
             markersize=5, linestyle='None')
    plt.show()


def plot_2D_compare(data_true, data_generated, title=None):
    fig = plt.figure()
    if title is not None:
        plt.title(f"{title}")
    plt.plot(data_true[:, 0], data_true[:, 1], marker='o', color='blue', alpha=0.4,
             markersize=5, linestyle='None', label='True samples')
    plt.plot(data_generated[:, 0], data_generated[:, 1], marker='o', color='red', alpha=0.4,
             markersize=5, linestyle='None', label='VAE samples')
    plt.legend()
    plt.show()


# 3D plot (interactive)
def plot_3D(dataset, title):
    n_max = 1500
    if len(dataset) > n_max:
        dataset = dataset[:n_max]
    else:
        dataset = dataset[:]
    data = pd.DataFrame(dataset.numpy())
    fig = px.scatter_3d(data, x=0, y=1, z=2, title=title)

    fig.show()


def plot_3D_(dataset, title):
    # Enabling the `widget` backend.
    # This requires jupyter-matplotlib a.k.a. ipympl.
    # ipympl can be installed via pip or conda.

    # %matplotlib widget
    n_max = 500
    if len(dataset) > n_max:
        dataset = dataset[:n_max]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if title is not None:
        plt.title(f"{title}")

    xs = dataset[:, 0]
    ys = dataset[:, 1]
    zs = dataset[:, 2]
    ax.scatter(xs, ys, zs, c='b')

    plt.show()


def plot_3D_compare(true_data, fake_data, recon_data=None, title=None):
    n_max = 250
    if len(true_data) > n_max:
        true_data = true_data[:n_max]
        fake_data = fake_data[:n_max]
        if recon_data is not None:
            recon_data = recon_data[:n_max]

    true_data = pd.DataFrame(true_data)
    true_data['trueVSfake'] = 'true'

    fake_data = pd.DataFrame(fake_data)
    fake_data['trueVSfake'] = 'fake'

    if recon_data is None:
        data = pd.concat([true_data, fake_data], ignore_index=True)
    else:
        recon_data = pd.DataFrame(recon_data)
        recon_data['trueVSfake'] = 'reconstructed'
        data = pd.concat([true_data, fake_data, recon_data], ignore_index=True)

    fig = px.scatter_3d(data, x=0, y=1, z=2, color='trueVSfake', title=title)
    fig.show()


def plot_true_vs_fake(vae, true_data, config, epoch=None):
    # sample test data from latent distribution
    latents = torch.randn((len(true_data), config.latent_dims)).to(config.device)
    sample_data = vae.decode(latents)

    recon_data = vae(true_data)[0].detach().numpy()

    if torch.is_tensor(true_data):
        true_data = true_data.cpu().detach().numpy()
    sample_data = sample_data.cpu().detach().numpy()

    if epoch is not None:
        title = f"Epoch {epoch}"
    else:
        title = None

    if config.data_dims == 2:
        plot_2D_compare(true_data, sample_data, title)
    if config.data_dims == 3:
        plot_3D_compare(true_data, sample_data, recon_data, title)


def plot_latent_violin(vae, train_dataloader, config, threshold=0.1):
    if config.dataset_type == "MNIST":  # use one batch
        (dataset, _) = next(iter(train_dataloader))
        dataset = dataset.view(-1, config.data_dims).to(config.device)
    else:
        dataset = next(iter(train_dataloader)).to(config.device)
    mu, logvar = vae.encode(dataset)

    mu = mu.to('cpu').detach().numpy()
    logvar = logvar.to('cpu').detach().numpy()

    fig, (ax_mu, ax_logvar) = plt.subplots(2, 1, constrained_layout=True,
                                           figsize=(0.45 * config.latent_dims + 2, 8))

    # visualize mean
    ax_mu.set_title('Encoder mean $\\nu_\Phi(x)$')
    ax_mu.violinplot(mu, showmeans=True)
    ax_mu.axhline(y=0.0, color='r', linestyle='dashed',
                  linewidth=2, dash_capstyle='round')
    ax_mu.axhline(y=-threshold, color='gray', linestyle='dotted',
                  linewidth=1)
    ax_mu.axhline(y=+threshold, color='gray', linestyle='dotted',
                  linewidth=1)

    # visualize variances
    ax_logvar.set_title('Encoder variance $\\tau^2_\Phi(x)$')
    ax_logvar.violinplot(np.exp(logvar), showmeans=True)

    if config.VAE_type != "VAE-2":
        ax_logvar.axhline(y=1.0, color='r', linestyle='dashed',
                          linewidth=2, dash_capstyle='round')
        ax_logvar.axhline(y=1.0 - threshold, color='gray', linestyle='dashed',
                          linewidth=1)
        ax_logvar.axhline(y=1.0 + threshold, color='gray', linestyle='dashed',
                          linewidth=1)

    elif config.VAE_type == "VAE-2":
        alpha2 = torch.exp(vae.logalpha2).data.cpu().numpy()
        xmin = [k + 0.55 for k in range(config.latent_dims)]
        xmax = [k + 1.45 for k in range(config.latent_dims)]
        plt.hlines(alpha2, xmin, xmax, color='r', linestyle='solid', linewidth=2)
        plt.hlines(alpha2 - threshold, xmin, xmax, color='gray', linestyle='dotted', linewidth=1)
        plt.hlines(alpha2 + threshold, xmin, xmax, color='gray', linestyle='dotted', linewidth=1)

    ax_logvar.set_ylim([0, None])

    # set style for the axes
    labels = [k + 1 for k in range(config.latent_dims)]
    for ax in [ax_mu, ax_logvar]:
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('latent dimensions')

    # Log plot object (as image)
    fig.savefig(f'{config.logdir}/Latent_Distribution_final.png')
    # plt.show()
    plt.close()


def plot_latent_space_2D(vae, data, epoch, logdir):  # writer, n_iter):
    """
    Plot distributions in latent space, i.e., mixture of Gaussians (enc_mu(x), enc_var(x)).
    """
    n_max = 1000
    if len(data) > n_max:
        data = data[:n_max]

    _, enc_mu, enc_logvar = vae(data)

    enc_mu = enc_mu.detach()
    enc_logvar = enc_logvar.detach().exp()
    n_kernels = len(enc_mu)

    distributions = []
    for i in range(n_kernels):
        distributions.append(multivariate_normal(enc_mu[i], enc_logvar[i]))

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-3.5, 3.5)
    ylim = (-3.5, 3.5)
    xres = 200
    yres = 200

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    zz = sum([k.pdf(xxyy) for k in distributions])

    # reshape and plot image
    img = zz.reshape((xres, yres))
    # writer.add_image("Latent Representation", img, n_iter)
    plt.figure(figsize=(8, 8))
    plt.imshow(img, interpolation="none", extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    plt.grid(False)
    plt.title(f"Latent Space representation after epoch {epoch}")
    # plt.savefig(f'{logdir}/latents_{epoch}.png')
    plt.savefig(f'{logdir}/Latent_2D_{epoch}.png')
    # plt.show()
    plt.close()


def plot_nu2_tau2(vae, data, epoch, logdir):  # writer, n_iter):
    """
    Plot dependency of nu^2 and tau^2 per latent (different color encoding).
    """
    n_max = 1000
    if len(data) > n_max:
        data = data[:n_max]

    _, enc_mu, enc_logvar = vae(data)

    enc_mu2 = enc_mu.cpu().detach() ** 2  # squared mean nu^2_h
    enc_logvar = enc_logvar.cpu().detach().exp()  # variance tau^2_h

    # define color map
    colors = cm.turbo(np.linspace(0, 1, vae.H))  # summer, cividis, tab20

    # initialize plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # plot "trade-off" circle
    x = np.linspace(0, 1, 400)
    ax.plot(x, np.sqrt(1 - x ** 2), '--', alpha=0.5, linewidth=2)

    # Use logarithmic scaling (if applicable)
    # ax.set_yscale('log')  # tau
    # ax.set_xscale('log')  # nu

    # plot mu^2 vs. tau^2 per latent & observation
    for h in range(vae.H):
        ax.scatter(enc_mu2[:, h], enc_logvar[:, h], color=colors[h], alpha=0.4, label=h + 1)

    # highlight the means
    for h in range(vae.H):
        ax.scatter(torch.mean(enc_mu2[:, h]), torch.mean(enc_logvar[:, h]), color=colors[h], edgecolor='black',
                   linewidth=2, marker="o", s=250, alpha=0.5)

    # legend with the unique colors from the scatter
    if vae.H <= 40:
        ax.legend(loc="upper right", title="Latents", bbox_to_anchor=(1.0, 1.0),
                  ncol=2, fancybox=True)
    ax.set_ylabel("$\\tau^2$ (Encoder variance)")
    ax.set_ylim(bottom=-0.1, top=1.3)
    ax.set_xlabel("$\\nu^2$ (Encoder mean)")
    ax.set_xlim(left=-0.1, right=2.6)

    plt.title(f"Latent Dependencies at epoch {epoch}")
    fig.savefig(f'{logdir}/latents_{epoch}.png')
    # plt.show()
    plt.close()


def create_embedding(data, labels, writer, n=100):
    assert len(data) == len(labels)

    # select random images and their target indices
    perm = torch.randperm(len(data))
    images, labels = data[perm][:n], labels[perm][:n]
    # get the class labels for each image
    # class_labels = [classes[lab] for lab in labels]
    # log embeddings
    features = images.view(-1, 28 * 28)

    writer.add_embedding(features,
                         metadata=labels,
                         label_img=images.unsqueeze(1))


def create_training_plot(bounds, bounds_held_out, three_entropies,
                         config):  # log_likelihood, test_log_likelihood, ground_truth_ll, ):
    """
    :param bounds: lower bound (per iteration)
    :param bounds_held_out: (per epoch)
    :param three_entropies: (per iteration)
    :param config: config file with name of data set (dataset_type),
                    name of the model (VAE_type) and save directory
    """
    bounds = np.asarray(bounds)
    bounds_held_out = np.asarray(bounds_held_out)
    three_entropies = np.asarray(three_entropies)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle(f'{config.VAE_type} ({config.dataset_type})')

    # upper plot (bound & 3 entropies)
    niter = range(1, len(three_entropies) + 1)
    ax1.plot(niter, bounds, label='lower bound', color='lightblue', zorder=1)
    ax1.plot(niter, three_entropies, label='three entropies', color='tab:green', zorder=4)
    ax1.set_xlim(0, len(three_entropies))
    ax2.set_ylim(bottom=bounds[-20:].min() * 0.9, top=bounds[-20:].max() * 1.05)
    ax1.grid(linestyle=':', linewidth=0.4)

    # different step size for epoch-wise plotting
    epoch_step = int(len(three_entropies) / len(bounds_held_out))
    n_epoch = range(0, len(three_entropies), epoch_step)
    ax1.plot(n_epoch, bounds_held_out, '--', label='lower bound (held-out)', color='tab:orange', alpha=0.5, zorder=3)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="lower right", ncol=2, prop={'size': 10})

    # lower plot (differences)
    diff = (bounds - three_entropies) / np.abs(bounds[-1]) * 100  # difference (in %)
    ax2.plot(niter, diff, color='lightblue', alpha=0.5, zorder=1)

    ax2.plot(niter, ema(diff, alpha=0.05), color='blue', zorder=2)

    bottom = - 10
    top = 10

    ax2.grid(linestyle=':', linewidth=0.4)
    ax2.set_xlim(0, len(three_entropies))
    ax2.set_ylim(bottom=bottom, top=top)
    ax2.set_ylabel(r'$\frac{\mathrm{LB} - \mathrm{3E}}{|\mathrm{LB}_\mathrm{final}|}$  [%]', fontsize=12)
    ax2.set_xlabel('Iterations')

    fig.tight_layout()
    fig.savefig(config.logdir + f'/{config.VAE_type}_{config.dataset_type}_plot.pdf')

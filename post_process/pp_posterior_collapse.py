import json

import matplotlib.pyplot as plt
import numpy as np
from attrdict import AttrDict
from matplotlib.ticker import MaxNLocator

from model import VariationalAutoencoder, VAE_MNIST
from model_CelebA import *
from prepare_data import get_dataloader
from utils import ema


def create_plot_posterior_collapse(PATH_TO_FOLDER, config, encoder_entropies, title):
    """
    Plot entropies of the encoder distributions & non-collapsed latent throughout training,
    add bar-plot of Encoder Entropies for fully trained model.
    """
    # load the encoder entropies saved during training
    H_enc_h = np.load(PATH_TO_FOLDER + '/H_enc_h_hist.npy')

    n_steps, n_latents = H_enc_h.shape
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1.25, 2]},
                                        figsize=[4, 7])  # figsize=[5.25, 4]
    # fig, ax1 = plt.subplots(figsize=[5, 4])  # just plots encoder entropies figsize=[4, 3.2]
    # fig.tight_layout()

    # fig = plt.figure()
    # ax1 = fig.add_subplot(2, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2)
    # Figure Suptitle
    plt.suptitle(title)
    # plt.title(title)
    niter_per_epoch = n_steps / config.epochs
    niter_epoch = [step / niter_per_epoch for step in range(n_steps)]

    # horizontal line indicating the threshold for posterior collapse and upper limit
    threshold = 1.0
    H_prior = np.log(2 * np.pi * np.e) / 2

    # custom grid lines
    ax1.set_axisbelow(True)
    ax1.grid(linestyle=':', linewidth=0.4, zorder=0)
    # ax.axhline(0.0, linestyle=':', color='black', alpha=0.3, linewidth=1.0)
    ax1.axhline(H_prior, linestyle=':', color='black', alpha=0.8, linewidth=1.5)
    ax1.axhline(H_prior, linestyle='-', color='black', alpha=0.2, linewidth=3.0)
    ax1.axhline(threshold, linestyle=':', color='tab:red', alpha=1.0, linewidth=1.5)
    ax1.axhline(threshold, linestyle='-', color='tab:red', alpha=0.2, linewidth=3.0)

    for latent in range(n_latents):
        ax1.plot(niter_epoch, ema(H_enc_h[:, latent], alpha=0.05),
                 alpha=0.7, linewidth=1.0)
        # alpha=0.5, linewidth=0.7)

    labels = [w.get_text() for w in ax1.get_yticklabels()]
    locs = list(ax1.get_yticks())
    # labels += [r'$\mathcal{H}^{prior}$']
    # labels += [r'$\frac{1}{2} \, \ln(2 \pi e)$']
    labels += [r'1.42']
    locs += [H_prior]
    if 1.5 in locs:
        labels.remove('1.5')
        locs.remove(1.5)
    ax1.set_yticks(locs)
    ax1.set_yticklabels(labels)

    ax1.set_xlim(0, config.epochs)
    ax1.set_ylabel('Avg. Encoder Entropy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylim(top=1.6)

    # middle plot
    ax2.grid(linestyle=':', linewidth=0.4, zorder=0)
    threshold = 1.0
    latents_non_collapsed = np.sum(H_enc_h < 1, axis=1)
    ax2.plot(niter_epoch, latents_non_collapsed, color='tab:green', alpha=1.0, zorder=2)
    ax2.set_xlim(0, config.epochs)
    ax2.set_ylabel('# Active Latents')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(bottom=0, top=n_latents + 1)

    # lower plot
    H = len(encoder_entropies)

    # ensure proper sorting
    values = -np.sort(- encoder_entropies)

    # horizontal line indicating the threshold for posterior collapse and upper limit
    threshold = 1.0
    H_prior = np.log(2 * np.pi * np.e) / 2

    # custom grid lines
    ax3.set_axisbelow(True)
    ax3.grid(linestyle=':', linewidth=0.4, zorder=0)
    # ax.axhline(0.0, linestyle=':', color='black', alpha=0.3, linewidth=1.0)
    ax3.axhline(H_prior, linestyle=':', color='black', alpha=0.8, linewidth=1.5)
    ax3.axhline(H_prior, linestyle='-', color='black', alpha=0.2, linewidth=3.0)
    ax3.axhline(threshold, linestyle=':', color='tab:red', alpha=1.0, linewidth=1.5)
    ax3.axhline(threshold, linestyle='-', color='tab:red', alpha=0.2, linewidth=3.0)
    # ax.axhline(ground_truth_ll, linestyle='--', color='black', label='groud-truth loglikelihood', zorder=5)

    # distinguish between collapsed and active
    n_collapsed = sum(values >= threshold)
    # plot collapsed latents
    ax3.bar(range(1, n_collapsed + 1), values[:n_collapsed], width=0.7, color='tab:red', alpha=1.0, zorder=2)
    # plot active latents
    ax3.bar(range(n_collapsed + 1, len(values) + 1), values[n_collapsed:], width=0.7, color='tab:green', alpha=1.0,
            zorder=2)

    # Add title and axis names
    # plt.title(r'(a) VAE-1 on CelebA ($\sigma = 1$)')
    # plt.title(title)
    ax3.set_xlabel('Latent dimensions')
    ax3.set_ylabel('Avg. Encoder Entropy')

    ax3.set_ylim(top=1.6)
    bar_width = ax3.patches[0].get_width()
    if H <= 20:
        ax3.set_xlim(1 - bar_width, H + bar_width)
    else:
        ax3.set_xlim(1 - 1.4 * bar_width, H + 1.4 * bar_width)
    plt.draw()
    labels = [w.get_text() for w in ax3.get_yticklabels()]
    locs = list(ax3.get_yticks())
    # labels += [r'$\mathcal{H}^{prior}$']
    # labels += [r'$\frac{1}{2} \, \ln(2 \pi e)$']
    labels += [r'1.42']
    locs += [H_prior]
    if 1.5 in locs:
        labels.remove('1.5')
        locs.remove(1.5)
    ax3.set_yticks(locs)
    ax3.set_yticklabels(labels)
    ax3.set_ylim(top=1.6)  # , bottom=min(values)-0.5)
    fig.tight_layout()
    # ax.grid(axis='y')
    # ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(top=0.95)  # 0.95
    fig.savefig(PATH_TO_FOLDER + f'/fig_encoder_entropies.pdf')
    # plt.show()


def get_final_encoder_entropies(PATH_TO_FOLDER, config, n_batches=10):
    """
    Simple function to calculate Encoder Entropies on n_batches batches on test data.
    :return: sorted list of averaged Encoder Entropies (descending).
    """
    config.device = 'cpu'  # switch to cpu calculations here (change if you like)

    # set up model class to load state dict
    if config.dataset_type == "MNIST":
        vae = VAE_MNIST(config=config).to(config.device)
    elif config.dataset_type == "CELEBA":
        vae = VanillaVAE(config=config).to(config.device)
    else:
        vae = VariationalAutoencoder(config=config).to(config.device)

    # load state dict and set to eval-mode
    vae.load_state_dict(torch.load(f"{PATH_TO_FOLDER}/vae"))
    vae.eval()

    _, test_dataloader = get_dataloader(config, test_shuffle=True)

    encoder_entropies = 0
    for n_iter, data_batch in enumerate(test_dataloader):
        # Preprocess batch (if necessary) & send to device
        if config.dataset_type == "MNIST":
            data_batch = data_batch[0]  # discard labels for MNIST
            data_batch = data_batch.view(-1, config.data_dims)  # flatten images
        data_batch = data_batch.to(config.device)

        # calculate ELBO & Sum of entropies
        _, enc_mu, enc_logvar = vae(data_batch)

        # encoding distributions
        # Encoder Distribution q(z|x) (as batch_size x latent_dims)
        q_z = Normal(enc_mu, (0.5 * enc_logvar).exp())
        H_enc = q_z.entropy().mean(0)  # mean over batch (same entropies anyway)

        encoder_entropies += H_enc
        if n_iter + 1 == n_batches:
            encoder_entropies = (encoder_entropies / (n_iter + 1)).detach().numpy()
            break
    # sort and save
    encoder_entropies = - np.sort(- encoder_entropies)
    np.savetxt(PATH_TO_FOLDER + '/encoder_entropies.txt', encoder_entropies)
    return encoder_entropies


def barplot_posterior_collapse(PATH_TO_FOLDER, encoder_entropies, title):
    # locally enable plotting of "H^prior"
    plt.rcParams.update({'mathtext.default': 'regular'})

    H = len(encoder_entropies)

    # ensure proper sorting
    values = -np.sort(- encoder_entropies)

    fig, ax = plt.subplots(1, 1, figsize=[4, 2.75])

    # horizontal line indicating the threshold for posterior collapse and upper limit
    threshold = 1.0
    H_prior = np.log(2 * np.pi * np.e) / 2

    # custom grid lines
    ax.set_axisbelow(True)
    ax.grid(linestyle=':', linewidth=0.4, zorder=0)
    # ax.axhline(0.0, linestyle=':', color='black', alpha=0.3, linewidth=1.0)
    ax.axhline(H_prior, linestyle=':', color='black', alpha=0.8, linewidth=1.5)
    ax.axhline(H_prior, linestyle='-', color='black', alpha=0.2, linewidth=3.0)
    ax.axhline(threshold, linestyle=':', color='tab:red', alpha=1.0, linewidth=1.5)
    ax.axhline(threshold, linestyle='-', color='tab:red', alpha=0.2, linewidth=3.0)
    # ax.axhline(ground_truth_ll, linestyle='--', color='black', label='groud-truth loglikelihood', zorder=5)

    # distinguish between collapsed and active
    n_collapsed = sum(values >= threshold)
    # plot collapsed latents
    ax.bar(range(1, n_collapsed + 1), values[:n_collapsed], width=0.7, color='tab:red', alpha=1.0, zorder=2)
    # plot active latents
    ax.bar(range(n_collapsed + 1, len(values) + 1), values[n_collapsed:], width=0.7, color='tab:green', alpha=1.0,
           zorder=2)

    # Add title and axis names
    # plt.title(r'(a) VAE-1 on CelebA ($\sigma = 1$)')
    plt.title(title)
    plt.xlabel('Latent dimensions')
    plt.ylabel('Avg. Encoder Entropy (per latent)')

    ax.set_ylim(top=1.6)
    bar_width = ax.patches[0].get_width()
    if H <= 20:
        ax.set_xlim(1 - bar_width, H + bar_width)
    else:
        ax.set_xlim(1 - 1.4 * bar_width, H + 1.4 * bar_width)
    plt.draw()
    labels = [w.get_text() for w in ax.get_yticklabels()]
    locs = list(ax.get_yticks())
    # labels += [r'$\mathcal{H}^{prior}$']
    # labels += [r'$\frac{1}{2} \, \ln(2 \pi e)$']
    labels += [r'1.42']
    locs += [H_prior]
    if 1.5 in locs:
        labels.remove('1.5')
        locs.remove(1.5)
    ax.set_yticks(locs)
    ax.set_yticklabels(labels)
    ax.set_ylim(top=1.6)  # , bottom=min(values)-0.5)

    # ax.grid(axis='y')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    fig.subplots_adjust(top=0.91)
    plt.savefig(PATH_TO_FOLDER + f'/fig_posterior_collapse.pdf')
    # plt.show()

# get the encoder entropies

# encoder_entropies = np.array([0.8, 1.1, 1.42, -1.1, -2, 1.05, -3, -2.3, 0.1, 0.4, -1.9,  -1.1, -2, -3, -2.3, 0.1, 0.9, -1.9, -4.7])
# create the plot
# encoder_entropies = np.loadtxt(PATH_TO_FOLDER + '/encoder_entropies.txt')
# barplot_posterior_collapse(encoder_entropies, ...)

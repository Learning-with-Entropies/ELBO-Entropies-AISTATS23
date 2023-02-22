"""
Script to (re-)produce the plots for the paper.
"""

import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import ema

matplotlib.use('TkAgg')


def create_plot_bounds_training(PATH_TO_FOLDER, config, title, lower_plot="relative"):
    """
    Plot ELBO and SumOfEntropies are compared throughout training.
    Either with close-up to visualize the noisy estimates with the ELBO,
    or with relative differences between ELBO and the three entropies.
    """

    # read in the files
    bounds = np.asarray(np.loadtxt(PATH_TO_FOLDER + '/bounds.txt'))
    bounds_held_out = np.asarray(np.loadtxt(PATH_TO_FOLDER + '/bounds_held_out.txt'))
    three_entropies = np.asarray(np.loadtxt(PATH_TO_FOLDER + '/three_entropies.txt'))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1.25]},
                                   figsize=[5.25, 4])  # figsize=[5.25, 4]
    # Figure Subtitleget_figsize
    fig.suptitle(title)

    # upper plot (bound & 3 entropies)
    niter = range(1, len(three_entropies) + 1)
    niter_per_epoch = len(three_entropies) / config.epochs
    niter_epoch = [iter / niter_per_epoch for iter in niter]
    # if bounds_held is not None:
    #     ax.plot(niter, log_likelihood, label='loglikelihood', color='tab:red', zorder=2)
    ax1.plot(niter_epoch, bounds, label='lower bound', color='lightblue', zorder=1)
    # if test_log_likelihood is not None:
    #     ax.plot(niter, test_log_likelihood, '--', label='held-out loglikelihood', color='tab:orange', zorder=3)
    ax1.plot(niter_epoch, three_entropies, label='three entropies', color='tab:green', zorder=4)
    # if ground_truth_ll is not None:
    #     ax.axhline(ground_truth_ll, linestyle='--', color='black', label='groud-truth loglikelihood', zorder=5)
    # ax1.set_ylim(0, len(three_entropies))
    ax1.set_xlim(0, config.epochs)
    ax1.grid(linestyle=':', linewidth=0.4)

    # different step size for epoch-wise plotting
    # epoch_step = int(len(three_entropies) / len(bounds_held_out))
    # n_epoch = range(epoch_step // config.n_epochs_log, len(three_entropies) + epoch_step // config.n_epochs_log,
    #                 epoch_step)
    n_epoch = range(1, config.epochs + 1, config.n_epochs_log)
    ax1.plot(n_epoch, bounds_held_out, '--', label='lower bound (held-out)', color='tab:orange', alpha=0.5, zorder=3)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="lower right", ncol=2, prop={'size': 7})  # 'size': 10
    if config.dataset_type in ["SUSY", "manifold"]:
        ax1.set_ylim(bottom=-bounds[-20:].max() * 1.5, top=bounds[-20:].max() * 1.5)
    # if config.dataset_type == "SUSY":
    # elif config.dataset_type == "CELEBA":
    #     ax1.set_ylim(bottom=-1000, top=18000)
    # ax1.set_xlim(left=-0.1, right=2.6)

    if lower_plot == "close-up":
        # lower plot (close-up)
        ax2.plot(niter_epoch, bounds, label='lower bound', color='lightblue', zorder=1)
        ax2.plot(niter_epoch, three_entropies, label='three entropies', color='tab:green', zorder=4)
        ax2.plot(n_epoch, bounds_held_out, '--', label='lower bound (held-out)', color='tab:orange', alpha=0.5,
                 zorder=3)

        handles, labels = ax2.get_legend_handles_labels()
        # ax2.set_xlim(0, len(three_entropies))
        # adapt close-up if applicable
        ax2.set_ylim(bottom=bounds[-20:].min() * 0.9, top=bounds[-20:].max() * 1.05)

        ax2.grid(linestyle=':', linewidth=0.4)
        # ax2.set_ylabel(r'$\frac{\mathrm{LB} - \mathrm{3E}}{|\mathrm{LB}_\mathrm{final}|}$  [%]', fontsize=12)
        ax2.set_xlabel('Epoch')
        # handles, labels = ax2.get_legend_handles_labels()
        # ax2.legend(handles, labels, loc="upper right", ncol=1, prop={'size': 10});
    elif lower_plot == "relative":
        # lower plot (differences)
        diff = (bounds - three_entropies) / np.abs(bounds[-1]) * 100  # difference (in %)
        ax2.plot(niter_epoch, diff, color='lightblue', alpha=0.5, zorder=1)

        # exponential moving average of
        # diff_ema = [diff[0]]
        # smooting = 0.005
        # for i in range(len(diff) - 1):
        #     diff_ema.append(smooting * diff[i] + (1 - smooting) * diff_ema[-1])
        ax2.plot(niter_epoch, ema(diff, alpha=0.01), color='blue', zorder=2)

        bottom = - 15
        top = 15

        ax2.grid(linestyle=':', linewidth=0.4)
        ax2.set_xlim(0, config.epochs)
        ax2.set_ylim(bottom=bottom, top=top)
        ax2.set_ylabel(r'$\frac{\mathrm{LB} - \mathrm{3E}}{|\mathrm{LB}_\mathrm{final}|}$  [%]', fontsize=12)
        ax2.set_xlabel('Epoch')
        # handles, labels = ax2.get_legend_handles_labels()
        # ax2.legend(handles, labels, loc="upper right", ncol=1, prop={'size': 10});
    else:
        raise NotImplementedError(f"Lower plot specification {lower_plot} unknown.")
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    # fig.subplots_adjust(bottom=0.15)
    fig.savefig(PATH_TO_FOLDER + f'/fig_bounds_full.pdf')


def read_files(PATH_TO_FOLDER, n_samples, n_runs):
    # read in a csv files for each sample size in the list n_samples
    ELBO_values = np.zeros([len(n_samples), n_runs])
    SumOfEntropies_values = np.zeros([len(n_samples), n_runs])
    for i, n in enumerate(n_samples):
        df = pd.read_csv(PATH_TO_FOLDER + f"/estimates_{n_samples[i]}.csv")
        ELBO_values[i] = df["ELBO"].to_numpy()
        SumOfEntropies_values[i] = df["H_sum"].to_numpy()
    return ELBO_values, SumOfEntropies_values


def create_plot_noise(PATH_TO_FOLDER, n_samples_list, n_runs, title):
    """
    Script to produce the plots in which ELBO and SumOfEntropies are compared for a pre-trained model.
    Comparison for different sample sizes, given in n_samples_list.
    Please make sure the corresponding files exist, i.e., have been generated by "pp_compare_ELBO_Entropies.py".
    :param n_samples_list:
    :param n_runs:
    """
    # read in the files estimates_XX in the model directory (if applicable)
    ELBO_values, SumOfEntropies_values = read_files(PATH_TO_FOLDER, n_samples_list, n_runs)

    # prepare plot
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1.25]}, figsize=[4, 3])  # 6, 4

    ax1.grid(linestyle=':', linewidth=0.4, zorder=0)
    ax2.grid(linestyle=':', linewidth=0.4, zorder=0)
    for i, n in enumerate(n_samples_list):
        ax1.scatter([i] * n_runs, ELBO_values[i], color='lightblue', alpha=1.0, label='lower bound', zorder=2)
        ax1.scatter([i] * n_runs, SumOfEntropies_values[i], color='tab:green', alpha=1.0, label='three entropies',
                    zorder=3)
        ax2.bar(i - 0.15, ELBO_values[i].std(), width=0.3, align='center', color='lightblue', alpha=1.0, zorder=2)
        ax2.bar(i + 0.15, SumOfEntropies_values[i].std(), width=0.3, align='center', color='tab:green', alpha=1.0,
                zorder=3)
        # print(ELBO_values[i].std())
        # print(SumOfEntropies_values[i].std())
    ax1.axhline(ELBO_values[-1].mean(), linestyle=':', color='lightblue', alpha=0.8, zorder=0)
    ax1.axhline(SumOfEntropies_values[-1].mean(), linestyle=':', color='tab:green', alpha=0.7, zorder=0)

    # Add title and axis names
    fig.suptitle(title)
    plt.xlabel('Sample Size')
    # plt.ylabel('Values')
    # labels = [w.get_text() for w in ax.get_xticklabels()]
    locs = [i for i in range(len(n_samples_list))]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], labels[:2], loc="upper right", ncol=2, prop={'size': 7})
    ax1.set_xticks(locs)
    ax1.set_xticklabels(n_samples_list)
    ax2.set_yscale('log')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_xticks(locs)
    ax2.set_xticklabels(n_samples_list)
    ax2.set_yticks([100, 1, 0.01])
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    fig.savefig(PATH_TO_FOLDER + f'/fig_bounds_noise.pdf')
    # plt.show()

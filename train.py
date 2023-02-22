import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from elbo import lower_bound, logalpha2_opt, logsigma2_opt
from utils import tensorlist_to_numpy
from visualize import plot_true_vs_fake, plot_latent_space_2D, plot_nu2_tau2

pi = np.pi
e = np.e


def beta_annealing(current_iter, n_iter_total, config):
    """
    Linear KL annealing schedule (beta in the ELBO).
    :param current_iter:
    :param n_iter_total:
    :param config: config dictionary, containing initial value of beta
    :return: beta
    """
    end_interpolate = 0.3
    start_interpolate = 0.0
    iter_start = start_interpolate * n_iter_total
    iter_end = end_interpolate * n_iter_total
    if current_iter < iter_start:
        # use predefined beta
        return config.beta
    if current_iter >= iter_end:
        # use default beta end (as in original ELBO)
        return 1.0
    # linear interpolation
    slope = (1.0 - config.beta) / (iter_end - iter_start)
    offset = config.beta - slope * iter_start
    return slope * current_iter + offset


def train(vae, train_dataloader, test_dataloader, config, writer):
    """
    The training loop for our VAE
    :param vae: VAE instance to train
    :param train_dataloader: Provides training data
    :param test_dataloader: Provides test data
    :param config: config dictionary
    :param writer: Tensorboard Summary Writer
    :return: test_ELBO
    """
    if config.dataset_type == "MNIST":
        n_channel = 1
    elif config.dataset_type == "CELEBA":
        n_channel = 3

    # OPTIONAL: Define list of parameters with higher learning rate (cf. config).
    # if config.VAE_type == "VAE-1":
    fast_params = ['logsigma2']
    if config.VAE_type == "VAE-2":
        fast_params += ['logalpha2']

    # Initialize Optimizer (with resp. learning rates)
    opt = torch.optim.Adam([
        {'params': [p for n, p in vae.named_parameters() if n not in fast_params], 'lr': config.lr_regular},
        {'params': [p for n, p in vae.named_parameters() if n in fast_params], 'lr': config.lr_fast}
        # higher learning rate for sigma (& alpha)
    ])
    # scheduler = StepLR(opt, step_size=config.epochs//10, gamma=0.9)

    # Initialize lists for plotting
    bounds = []
    bounds_held_out = []
    three_entropies = []

    # Initialize Entropies History
    H_enc_h_hist = []  # individual per latent

    # Sample fixed test batch to track changes in entropies
    fixed_test_batch = next(iter(train_dataloader))
    # Preprocess batch (if necessary) & send to device
    if config.dataset_type == "MNIST":
        fixed_test_batch = fixed_test_batch[0]  # discard labels for MNIST
        fixed_test_batch = fixed_test_batch.view(-1, config.data_dims)  # flatten images
    fixed_test_batch = fixed_test_batch.to(config.device)

    # Training Loop
    print(f"Start training for {config.epochs} epochs...")

    n_iter_total = config.epochs * len(train_dataloader)
    for epoch in tqdm(range(config.epochs)):
        vae.train()
        for i, data_batch in enumerate(train_dataloader):
            n_iter = i + epoch * len(train_dataloader)

            # OPTIONAL: KL annealing schedule (cf. beta_annealing)
            if config.KL_annealing:
                beta = beta_annealing(n_iter, n_iter_total, config)
            else:
                beta = config.beta
            writer.add_scalar('beta', beta, n_iter)

            # Preprocess batch (if necessary) & send to device
            if config.dataset_type == "MNIST":
                data_batch = data_batch[0]  # discard labels for MNIST
                data_batch = data_batch.view(-1, config.data_dims)  # flatten images
            data_batch = data_batch.to(config.device)

            # Weight constraint via manual rescaling (for VAE-2)
            if config.VAE_type == "VAE-2" and config.weight_constraint == "rescale":
                with torch.no_grad():
                    vae.decoder_hidden.weight = nn.Parameter(
                        1 / torch.norm(vae.decoder_hidden.weight, dim=0) * vae.decoder_hidden.weight)

            # OPTIONAL: set optimal variances
            if config.optimal_vars:
                # Try full/part of data set vs. current batch
                # data_sigma_opt = train_dataloader.dataset.samples # [:10000]
                data_sigma_opt = data_batch
                recon_sigma_opt, enc_mu, enc_logvar = vae(data_sigma_opt)
                with torch.no_grad():
                    vae.logsigma2 = nn.Parameter(logsigma2_opt(vae, data_sigma_opt, recon_sigma_opt, config))
                    if config.VAE_type == "VAE-2":
                        vae.logalpha2 = nn.Parameter(logalpha2_opt(enc_mu, enc_logvar))

            # Calculate the Lower Bound (ELBO)
            recon_batch, enc_mu, enc_logvar = vae(data_batch)
            ELBO, log_prob, kl_div, H_prior, H_enc, H_dec = lower_bound(vae, data_batch, recon_batch, enc_mu,
                                                                        enc_logvar, config, beta=beta)
            # Calculate the Sum of Entropies
            H_sum = - H_prior - H_dec + H_enc

            # Weight Constraint via Regularization
            if config.VAE_type == "VAE-2" and config.weight_constraint == "regularize":
                gamma = 1
                penalty = torch.sum((torch.norm(vae.decoder_hidden.weight, dim=0) ** 2 - 1) ** 2)
                writer.add_scalar('W1_norms', penalty, n_iter)
                ELBO -= gamma * penalty

            # training step
            opt.zero_grad()
            (-ELBO).backward()
            opt.step()

            # do scheduler step
            # scheduler.step()

            # Logging
            # 1. ELBO & Entropies
            if beta != 1 or config.KL_annealing:
                writer.add_scalars('ELBO', {'ELBO': log_prob - kl_div,
                                            'H_sum': H_sum,
                                            'ELBO_beta': ELBO}, n_iter)
            else:
                writer.add_scalars('ELBO', {'ELBO': log_prob - kl_div,
                                            'H_sum': H_sum,
                                            'H_sum-ELBO': H_sum - (log_prob - kl_div)}, n_iter)
            writer.add_scalars('ELBO_parts', {'ELBO': log_prob - kl_div,
                                              'log_prob': log_prob,
                                              'kl_div': kl_div}, n_iter)

            # 2.1 Encoder entropies to detect & analyze posterior collapse
            # Estimate entropies on fixed test batch
            recon_batch_test, _, enc_logvar_test = vae(fixed_test_batch)
            H_enc_test = torch.mean(enc_logvar + np.log(2 * pi * e), dim=0) / 2  # per latent
            H_enc_h_hist.append(H_enc_test.detach())
            writer.add_scalars('Encoder Entropies (H_enc_h)', {f"{i}": H_enc_test[i] for i in range(vae.H)}, n_iter)
            # 2.2 Non-collapsed latents
            threshold = 1.0
            latents_non_collapsed = sum([1 for H_enc_h in H_enc_test if H_enc_h <= threshold])
            writer.add_scalar('Non-collapsed Latents', latents_non_collapsed, n_iter)

            bounds.append((log_prob - kl_div).cpu().detach().numpy())
            three_entropies.append(H_sum.cpu().detach().numpy())

        # Evaluation & Testing
        if epoch % config.n_epochs_log == 0:
            vae.eval()
            with torch.no_grad():
                if config.dataset_type in ["MNIST", "CELEBA"]:
                    # Evaluate Generation (8x8 images)
                    # Generate from Prior samples
                    if config.VAE_type != 'VAE-2':
                        # Sample from standard normal prior
                        latent_code = torch.randn(64, config.latent_dims).to(config.device)
                    elif config.VAE_type == 'VAE-2':
                        # Sample from learnable normal prior
                        latent_code = torch.exp(0.5 * vae.logalpha2) * torch.randn(64, config.latent_dims).to(
                            config.device)
                    sample = vae.decode(latent_code).cpu()

                    writer.add_images('samples_gen', sample.view(64, n_channel, config.img_dim, config.img_dim), n_iter)

                ELBO_held_out = 0
                for i, test_batch in enumerate(test_dataloader):
                    # preprocess if necessary
                    if config.dataset_type == "MNIST":
                        test_batch = test_batch[0]  # discard labels for MNIST
                        test_batch = test_batch.view(-1, config.data_dims)
                    test_batch = test_batch.to(config.device)
                    recon_test_batch, test_mu, test_logvar = vae(test_batch)
                    ELBO, log_prob, kl_div, H_prior, H_enc, H_dec = lower_bound(vae, test_batch, recon_test_batch,
                                                                                test_mu, test_logvar, config,
                                                                                beta=beta)
                    ELBO_held_out += log_prob - kl_div

                    # Evaluate Reconstruction
                    if i == 0 and epoch % config.n_epochs_vis == 0:
                        if config.dataset_type in ["MNIST", "CELEBA"]:
                            comparison = torch.cat(
                                [test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[:8],
                                 recon_test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[
                                 :8],
                                 test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[8:16],
                                 recon_test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[
                                 8:16],
                                 test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[16:24],
                                 recon_test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[
                                 16:24],
                                 test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[24:32],
                                 recon_test_batch.view(config.batch_size, n_channel, config.img_dim, config.img_dim)[
                                 24:32]])

                            writer.add_images('samples_rec', comparison.cpu(), n_iter)

                        elif config.data_dims in [2, 3]:
                            if config.latent_dims == 2:
                                test_data = test_dataloader.dataset.data
                            elif config.latent_dims == 3:
                                test_data = test_dataloader.dataset.dataset
                            plot_latent_space_2D(vae, test_data, epoch=epoch, logdir=config.logdir)
                            plot_true_vs_fake(vae, test_data, config, epoch=epoch)
                if epoch % config.n_epochs_vis == 0:
                    plot_nu2_tau2(vae, test_batch, epoch, logdir=config.logdir)

                ELBO_held_out = ELBO_held_out / len(test_dataloader)
                writer.add_scalar('test_ELBO', ELBO_held_out, n_iter)
                bounds_held_out.append(ELBO_held_out.cpu().detach().numpy())
    H_enc_h_hist = tensorlist_to_numpy(H_enc_h_hist)
    return bounds, bounds_held_out, three_entropies, H_enc_h_hist

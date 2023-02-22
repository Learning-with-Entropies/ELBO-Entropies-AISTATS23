import torch
import numpy as np
from torch.distributions import Normal, kl_divergence

pi = np.pi
e = np.e

def lower_bound(vae, x, x_recon, enc_mu, enc_logvar, config, beta=1.):
    """
    Calculate Evidence Lower Bound (ELBO) for given VAE for current batch
    :param vae: VAE instance
    :param x: input data (usually a batch)
    :param x_recon: reconstructed data (by VAE)
    :param enc_mu: means of x, predicted by Encoder net
    :param enc_logvar: log(variances) of x, predicted by Encoder net
    :param config: config dictionary
    :param beta: current beta
    :return:
        either ELBO, Recon Loss, KL Divergence, Entropies (training with ELBO)
        or     Sum of Entropies, Recon Loss, KL Divergence, Entropies (training with Entropies)
    """

    # Initialize Distributions
    # Encoder Distribution q(z|x)
    q_z = Normal(enc_mu, (0.5 * enc_logvar).exp())
    # Decoder Distribution p(x|z)
    p_x_z = Normal(x_recon, (0.5 * vae.logsigma2).exp())  # Gaussian is symmetric in x, x_recon
    if config.VAE_type == "VAE-2":
        # learnable prior covariance
        p_z = Normal(torch.zeros(vae.H).to(config.device), (0.5 * vae.logalpha2).exp())
    else:
        # standard normal prior
        p_z = Normal(torch.zeros(vae.H).to(config.device), torch.ones(vae.H).to(config.device))
    if config.dataset_type == "MNIST":
        x = x.view(-1, 784)

    # Calculate parts of the ELBO
    log_prob = p_x_z.log_prob(x).mean(0).sum()
    kl_div = kl_divergence(q_z, p_z).mean(0).sum()

    ELBO = log_prob - beta * kl_div

    # Calculate Three Entropies
    H_prior = p_z.entropy().sum()  # sum over latent dims
    H_enc = q_z.entropy().mean(0).sum()  # mean over batch, sum over latent dims
    H_dec = p_x_z.entropy().mean(0).sum()  # mean over batch, sum over data dims

    return ELBO, log_prob, kl_div, H_prior, H_enc, H_dec

def logsigma2_opt(vae, data_batch, recon_batch, config):
    """
    Calculate optimal Decoder Variance log(sigma_opt^2)
    :param vae: VAE instance
    :param data_batch:
    :param recon_batch:
    :param config:
    """
    if config.dataset_type == "CELEBA":
        # norm over all channels
        sigma2_opt = torch.mean(torch.norm(data_batch - recon_batch, p=2, dim=[1, 2, 3]) ** 2) / vae.D
    else:
        sigma2_opt = torch.mean(torch.norm(data_batch - recon_batch, p=2, dim=1) ** 2) / vae.D
    return torch.log(sigma2_opt)


def logalpha2_opt(enc_mu, enc_logvar):
    """
    Calculate optimal (learnable) prior variance,
    which is given as the second moment of the Encoder parameters.
    :param enc_mu: Encoder means
    :param enc_logvar: Encoder (log-)variances
    """
    alpha2_opt = torch.mean(enc_mu.pow(2) + (enc_logvar).exp(), dim=0)
    return torch.log(alpha2_opt)
